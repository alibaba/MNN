//
//  DeconvSingleInputExecution.cpp
//  MNN
//
//  Created by MNN on 2022/03/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "DeconvSingleInputExecution.hpp"
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"
namespace MNN {
namespace CUDA {
__global__ void DeconvKernelReorder(const float* B, half* BP, int kw, int kh, int ic, int oc, int icPack) {
    int kernelCount = kw * kh;
    int e = oc * kernelCount;
    int l = ic;
    int eDiv = UP_DIV(e, MATMULPACK);
    int eAlign = eDiv * MATMULPACK;
    int lDiv = UP_DIV(l, icPack);
    int lAlign = lDiv * icPack;

    int maxCount = eAlign * lAlign;

    for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {

        int lR = indexO % icPack;
        int tmp = indexO / icPack;
        int eR = tmp % MATMULPACK;
        int tmp2 = tmp / MATMULPACK;
        int lC = tmp2 % lDiv;
        int eC = tmp2 / lDiv;

        half* dst = BP + indexO;
        int sL = lC * icPack + lR;//ic_idx
        int sE = eC * MATMULPACK + eR;
        if (sL >= ic) {
            *dst = 0.0;
            continue;
        }

        int oEC = sE / (kernelCount);//oc_idx
        int oEk = sE % kernelCount;//khw_idx
        if (sE >= e) {
            *dst = 0.0;
            continue;
        }
        const float* src = B + sL * kernelCount * oc + oEk + oEC * kernelCount;
        *dst = *src;
    }
}

template<typename T>
__global__ void DeconvInputRerange(const int count,
        const InputReorderParameter* param,
        const T* Inp,
        __half* InpRe
        ) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        int l = 16 * param->lpack_size;
        int h = 16 *  param->hpack_size;
        int lIndex = i % l;
        int hIndex = i / l;
        int lU = lIndex / 16;
        int lR = lIndex % 16;
        int hU = hIndex / 16;
        int hR = hIndex % 16;

        __half* dst = InpRe + hU * param->lpack_size * 16 * 16 + lU * 16 * 16 + lR + hR * 16;

        if(hIndex >= param->h_size || lIndex >= param->l_size) {
            dst[0] = (__half)0.0;
            break;
        }
        const int channel_pack = ((param->l_size + 7) / 8) * 8;
        T value = Inp[hIndex * channel_pack + lIndex];
        dst[0] = (half)value;
    }
}

template <typename Dtype>
__global__ void Col2Im(const int n, const Dtype* data_col,
    const int batch, const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    const float* bias, Dtype* data_im
) {
    const int channel_pack = ((channels+7) / 8) * 8;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x) {
        Dtype val = 0;        
        const int c_p = index % channel_pack;
        const int idx_tmp = index / channel_pack;
        const int b_im = idx_tmp / (width * height);
        const int hw = idx_tmp % (width * height);
        const int c_im = c_p;
        const int w_im = hw % width + pad_w;
        const int h_im = hw / width + pad_h;

        if(c_im >= channels) {
            data_im[index] = val;
            break;
        }
        if(nullptr != bias) {
            val += (Dtype)bias[c_im];
        }
        int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
        int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
        // compute the start and end of the output
        const int w_col_start =
            (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
        const int w_col_end = min(w_im / stride_w + 1, width_col);
        const int h_col_start =
            (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
        const int h_col_end = min(h_im / stride_h + 1, height_col);
        // TODO: use LCM of stride and dilation to avoid unnecessary loops
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                int h_k = (h_im - h_col * stride_h);
                int w_k = (w_im - w_col * stride_w);
                if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                    h_k /= dilation_h;
                    w_k /= dilation_w;

                    const int data_col_index = ((((c_im * kernel_h + h_k) * kernel_w + w_k) * batch + b_im) *
                                            height_col + h_col) * width_col + w_col;
                    val += data_col[data_col_index];
                }
            }
        }


        data_im[index] = val;
    }
}


DeconvSingleInputExecution::Resource::Resource(Backend* bn, const MNN::Op* op) {
    mBackend = bn;
    auto runtime = static_cast<CUDABackend*>(bn)->getCUDARuntime();
    
    auto conv       = op->main_as_Convolution2D();
    auto common     = conv->common();
    mKernelInfo.kernelX        = common->kernelX();
    mKernelInfo.kernelY        = common->kernelY();
    mKernelInfo.groups         = common->group();
    mKernelInfo.strideX        = common->strideX();
    mKernelInfo.strideY        = common->strideY();
    mKernelInfo.dilateX        = common->dilateX();
    mKernelInfo.dilateY        = common->dilateY();
    mKernelInfo.activationType = common->relu() ? 1 : (common->relu6() ? 2 : 0);

    //weight host->device
    const float* filterDataPtr = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, conv, &filterDataPtr, &weightSize);
    mKernelInfo.kernelN = common->outputCount();
    mKernelInfo.kernelC = weightSize / mKernelInfo.kernelN / mKernelInfo.kernelX / mKernelInfo.kernelY;

    MatMulParam param;
    int e = mKernelInfo.kernelN * mKernelInfo.kernelX * mKernelInfo.kernelY;
    int l = mKernelInfo.kernelC;
    int h = 0;
    param.elh[0] = e;
    param.elh[1] = l;
    param.elh[2] = h;
    param.elhPack[0] = UP_DIV(e, 16);
    param.elhPack[1] = UP_DIV(l, 16);
    param.elhPack[2] = UP_DIV(h, 16);

    param.aStride[0] = 1;
    param.aStride[1] = e;
    param.aStride[2] = 0;
    param.bStride[0] = 0;
    param.bStride[1] = h;
    param.bStride[2] = 1;

    auto gpuParam = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(sizeof(MatMulParam));
    auto tempCacheBuffer = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(weightSize * sizeof(float));
    float* cacheWeight = (float*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
    runtime->memcpy(cacheWeight, filterDataPtr, weightSize * sizeof(float), MNNMemcpyHostToDevice);
    runtime->memcpy((uint8_t*)gpuParam.first + gpuParam.second, &param, sizeof(MatMulParam), MNNMemcpyHostToDevice);
    
    // Reorder weight
    weightTensor.reset(Tensor::createDevice<int16_t>({param.elhPack[0] * param.elhPack[1] * (MATMULPACK * MATMULPACK)}));
    bn->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
    mFilter = (void *)weightTensor.get()->buffer().device;    
    
    auto& prop = runtime->prop();
    int cores = prop.multiProcessorCount;
    int threadNumbers = prop.maxThreadsPerBlock;
    DeconvKernelReorder<<<cores, threadNumbers>>>((float*)cacheWeight, (half*)mFilter,
        mKernelInfo.kernelX, mKernelInfo.kernelY, mKernelInfo.kernelC, mKernelInfo.kernelN, MATMULPACK);
    
    static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempCacheBuffer);
    static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(gpuParam);

    // Copy Bias
    int biasSize = conv->bias()->size();
    biasTensor.reset(Tensor::createDevice<float>({biasSize}));
    bn->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
    mBias = (void *)biasTensor.get()->buffer().device;
    cuda_check(cudaMemcpy(mBias, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));
    
}

DeconvSingleInputExecution::Resource::~Resource() {
    // Do nothing
}
DeconvSingleInputExecution::DeconvSingleInputExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res) : Execution(backend), mOp(op) {
    mResource = res;
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    auto staticPool = static_cast<CUDABackend*>(backend)->getStaticBufferPool();
    mGpuMatMulParam = staticPool->alloc(sizeof(MatMulParam));
    mGpuCol2ImParam = staticPool->alloc(sizeof(Col2ImParameter));
    mGpuInpReorderParam = staticPool->alloc(sizeof(InputReorderParameter));
}

DeconvSingleInputExecution::~DeconvSingleInputExecution() {
    auto staticPool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
    staticPool->free(mGpuMatMulParam);
    staticPool->free(mGpuCol2ImParam);
    staticPool->free(mGpuInpReorderParam);
}
bool DeconvSingleInputExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new DeconvSingleInputExecution(bn, op, mResource);
    *dst = dstExe;
    return true;
}


ErrorCode DeconvSingleInputExecution::onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto input = inputs[0], output = outputs[0];
    const int UNIT = 1;
    auto convCommon = mOp->main_as_Convolution2D()->common();

    // Input Rerange Param
    mInpReorderParameter.hw_size = input->height() * input->width();
    mInpReorderParameter.ic_stride = mInpReorderParameter.hw_size;
    mInpReorderParameter.ib_stride = mInpReorderParameter.hw_size * input->channel();
    mInpReorderParameter.oc_stride = mInpReorderParameter.ib_stride;
    mInpReorderParameter.ob_stride = mInpReorderParameter.hw_size;
    mInpReorderParameter.l_size    = input->channel();
    mInpReorderParameter.h_size    = input->batch() * mInpReorderParameter.hw_size;
    mInpReorderParameter.lpack_size = UP_DIV(mInpReorderParameter.l_size, 16);
    mInpReorderParameter.hpack_size = UP_DIV(mInpReorderParameter.h_size, 16);

    runtime->memcpy((uint8_t*)mGpuInpReorderParam.first + mGpuInpReorderParam.second, &mInpReorderParameter, sizeof(InputReorderParameter), MNNMemcpyHostToDevice);

    // Col2Im Param
    auto pad = ConvolutionCommon::convolutionTransposePad(input, output, mOp->main_as_Convolution2D()->common());
    mCol2ImParamter.dilateX         = convCommon->dilateX();
    mCol2ImParamter.dilateY         = convCommon->dilateY();
    mCol2ImParamter.strideX         = convCommon->strideX();
    mCol2ImParamter.strideY         = convCommon->strideY();
    mCol2ImParamter.ic              = input->channel();
    mCol2ImParamter.oc              = output->channel();
    mCol2ImParamter.kernelX         = convCommon->kernelX();
    mCol2ImParamter.kernelY         = convCommon->kernelY();
    mCol2ImParamter.padX = pad.first;
    mCol2ImParamter.padY = pad.second;

    mCol2ImParamter.ih = input->height();
    mCol2ImParamter.iw = input->width();
    mCol2ImParamter.oh = output->height();
    mCol2ImParamter.ow = output->width();
    mCol2ImParamter.ob = output->batch();

    runtime->memcpy((uint8_t*)mGpuCol2ImParam.first + mGpuCol2ImParam.second, &mCol2ImParamter, sizeof(Col2ImParameter), MNNMemcpyHostToDevice);

    // Matmul Param
    int e = output->channel() * mCol2ImParamter.kernelX * mCol2ImParamter.kernelY;
    int l = input->channel();
    int h = input->height() * input->width() * output->batch();

    mMatMulParam.elh[0] = e;
    mMatMulParam.elh[1] = l;
    mMatMulParam.elh[2] = h;
    mMatMulParam.elhPack[0] = UP_DIV(e, 16);
    mMatMulParam.elhPack[1] = UP_DIV(l, 16);
    mMatMulParam.elhPack[2] = UP_DIV(h, 16);

    mMatMulParam.bStride[0] = 0;
    mMatMulParam.bStride[1] = input->height() * input->width();
    mMatMulParam.bStride[2] = 1;

    mMatMulParam.cStride[0] = h;
    mMatMulParam.cStride[1] = 0;
    mMatMulParam.cStride[2] = 1;
    mMatMulParam.aPStride[0] = 256 * mMatMulParam.elhPack[1];
    mMatMulParam.aPStride[1] = 256;
    mMatMulParam.aPStride[2] = 16;
    mMatMulParam.bPStride[0] = 256 * mMatMulParam.elhPack[1];
    mMatMulParam.bPStride[1] = 256;
    mMatMulParam.bPStride[2] = 16;

    if (convCommon->relu()) {
        mMatMulParam.minValue = 0.0f;
    }
    if (convCommon->relu6()) {
        mMatMulParam.minValue = 0.0f;
        mMatMulParam.maxValue = 6.0f;
    }
    runtime->memcpy((uint8_t*)mGpuMatMulParam.first + mGpuMatMulParam.second, &mMatMulParam, sizeof(MatMulParam), MNNMemcpyHostToDevice);

    // Alloc temp cuda memory
    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    auto buffer1 = pool->alloc(sizeof(float) * mMatMulParam.elhPack[0] * mMatMulParam.elhPack[2]* MATMULPACK * MATMULPACK);
    auto buffer2 = pool->alloc(sizeof(__half) * mMatMulParam.elhPack[1] * mMatMulParam.elhPack[2] * MATMULPACK * MATMULPACK);

    mIm2ColBuffer = (float*)((uint8_t*)buffer1.first + buffer1.second);
    mInputBuffer = (__half*)((uint8_t*)buffer2.first + buffer2.second);

    pool->free(buffer2);
    pool->free(buffer1);

    return NO_ERROR;
}

ErrorCode DeconvSingleInputExecution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    //MNN_PRINT("cuda convSingleInput onExecute in, inputsize:%d %d\n", (int)inputs.size(), workspace_size_);
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]);

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    const void *input_addr = (const void*)inputs[0]->deviceId();
    const void *filter_addr = mResource->mFilter;
    const void *bias_addr = mResource->mBias;
    void *output_addr = (void*)outputs[0]->deviceId();

    auto gpuInpReorder = (const InputReorderParameter*)((uint8_t*)mGpuInpReorderParam.first + mGpuInpReorderParam.second);
    auto gpuCol2Im = (const Col2ImParameter*)((uint8_t*)mGpuCol2ImParam.first + mGpuCol2ImParam.second);
    auto gpuMatMul = (const MatMulParam*)((uint8_t*)mGpuMatMulParam.first + mGpuMatMulParam.second);

    const int rerangeCount = mInpReorderParameter.lpack_size * mInpReorderParameter.hpack_size * 16 * 16;
    int inp_block_num = runtime->blocks_num(rerangeCount);
    int inp_thread_num = runtime->threads_num();

    // Do input Rerange
    //runtime->memset(mInputBuffer, 0, mMatMulParam.elhPack[2] * mMatMulParam.elhPack[1] * MATMULPACK * MATMULPACK * sizeof(__half));
    if(bytes == 4) {
        DeconvInputRerange<<<inp_block_num, inp_thread_num>>>(rerangeCount, gpuInpReorder, (const float*)input_addr, mInputBuffer);
        // Do Gemm operation 
        GemmPackedMain(runtime, &mMatMulParam, gpuMatMul, (float*)mIm2ColBuffer, (const half*)filter_addr, (const half*)mInputBuffer, nullptr, bytes, false, false);
    } else {
        DeconvInputRerange<<<inp_block_num, inp_thread_num>>>(rerangeCount, gpuInpReorder, (const half*)input_addr, mInputBuffer);
        // Do Gemm operation 
        GemmPackedMain(runtime, &mMatMulParam, gpuMatMul, (half*)mIm2ColBuffer, (const half*)filter_addr, (const half*)mInputBuffer, nullptr, bytes, false, false);

    }

    // Do Col2Im trans
    int height_col = mCol2ImParamter.ih;
    int width_col = mCol2ImParamter.iw;
    int num_kernels = mCol2ImParamter.ob * UP_DIV(mCol2ImParamter.oc, 8) * mCol2ImParamter.oh * mCol2ImParamter.ow * 8;

    int col2im_block_num = runtime->blocks_num(num_kernels);
    int col2im_thread_num = runtime->threads_num();

    // printf("col2im:%d, %d-%d-%d-%d-%d-%d\n %d-%d-%d-%d-%d-%d\n %d-%d\n", mCol2ImParamter.ob, mCol2ImParamter.oh, mCol2ImParamter.ow, mCol2ImParamter.oc, \
    //     mCol2ImParamter.ih, mCol2ImParamter.iw, mCol2ImParamter.ic, \
    //     mCol2ImParamter.padX, mCol2ImParamter.padY, mCol2ImParamter.kernelX, mCol2ImParamter.kernelY, mCol2ImParamter.strideX, mCol2ImParamter.strideY, \
    //     col2im_block_num, col2im_thread_num);
    
    if(bytes == 4) {
        Col2Im<float><<<col2im_block_num, col2im_thread_num>>>(
            num_kernels, (const float*)mIm2ColBuffer, mCol2ImParamter.ob, mCol2ImParamter.oh, mCol2ImParamter.ow, mCol2ImParamter.oc, 
            mCol2ImParamter.kernelY, mCol2ImParamter.kernelX, mCol2ImParamter.padY, mCol2ImParamter.padX, 
            mCol2ImParamter.strideY, mCol2ImParamter.strideX, mCol2ImParamter.dilateY, mCol2ImParamter.dilateX,
            height_col, width_col, (const float*)bias_addr, (float *)output_addr);
    } else {
        Col2Im<half><<<col2im_block_num, col2im_thread_num>>>(
            num_kernels, (const half*)mIm2ColBuffer, mCol2ImParamter.ob, mCol2ImParamter.oh, mCol2ImParamter.ow, mCol2ImParamter.oc, 
            mCol2ImParamter.kernelY, mCol2ImParamter.kernelX, mCol2ImParamter.padY, mCol2ImParamter.padX, 
            mCol2ImParamter.strideY, mCol2ImParamter.strideX, mCol2ImParamter.dilateY, mCol2ImParamter.dilateX,
            height_col, width_col, (const float*)bias_addr, (half *)output_addr);
    }

    return NO_ERROR;
}

class CUDADeconvolutionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, 
            const MNN::Op* op, Backend* backend) const override {
        if (nullptr != op->main_as_Convolution2D()->quanParameter()) {
            auto quan = op->main_as_Convolution2D()->quanParameter();
            if (1 == quan->type() || 2 == quan->type()) {
                MNN_PRINT("cuda Deconv quant type 1 or 2 not support\n");
                return nullptr;
            }
        }

        if(inputs.size() == 3) {
            MNN_PRINT("Deconv inputs size:3 not support\n");
            return nullptr;
        } else if(inputs.size() == 1) {
            std::shared_ptr<DeconvSingleInputExecution::Resource> resource(new DeconvSingleInputExecution::Resource(backend, op));
            return new DeconvSingleInputExecution(backend, op, resource);
        } else {
            MNN_PRINT("Deconv inputs size:%d not support", (int)inputs.size());
            return nullptr;
        }
    }
};

CUDACreatorRegister<CUDADeconvolutionCreator> __DeConvExecution(OpType_Deconvolution);

}// namespace CUDA
}// namespace MNN
