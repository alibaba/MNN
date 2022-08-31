//
//  ConvSingleInputExecution.cpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvSingleInputExecution.hpp"
#include "ConvWinogradExecution.hpp"
#include "ConvCutlassExecution.hpp"
#include "Raster.cuh"
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"

// 16 / sizeof(int4)
namespace MNN {
namespace CUDA {

__global__ void KernelReorder(const float* B, half* BP, int kw, int kh, int ic, int oc, int ocPack) {
    int icC4 = UP_DIV(ic, PACK_NUMBER);
    int kernelCount = kw * kh;
    int l = icC4 * kernelCount * PACK_NUMBER;
    int h = oc;
    int lDiv = UP_DIV(l, MATMULPACK);
    int lAlign = lDiv * MATMULPACK;
    int hAlign = UP_DIV(h, ocPack) * ocPack;
    int maxCount = hAlign * lAlign;

    for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
        int lR = indexO % MATMULPACK;
        int tmp = indexO / MATMULPACK;
        int hR = tmp % ocPack;
        int tmp2 = tmp / ocPack;
        int lC = tmp2 % lDiv;
        int hC = tmp2 / lDiv;
        half* dst = BP + indexO;
        int sH = hC * ocPack + hR;
        int sL = lC * MATMULPACK + lR;
        if (sH >= oc) {
            *dst = 0.0;
            continue;
        }
        int sLR = sL % PACK_NUMBER;
        int sLC = sL / PACK_NUMBER;
        int iLC = sLC / (kernelCount);
        int ik = sLC % kernelCount;
        int iz = iLC * PACK_NUMBER + sLR;
        if (iz >= ic) {
            *dst = 0.0;
            continue;
        }
        const float* src = B + sH * kernelCount * ic + ik + iz * kernelCount;
        *dst = *src;
    }
}

ConvSingleInputExecution::Resource::Resource(Backend* bn, const MNN::Op* op) {
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
    int icDiv = UP_DIV(mKernelInfo.kernelC, PACK_NUMBER);

    MatMulParam param;
    int e = 0;
    int l = mKernelInfo.kernelX * mKernelInfo.kernelY * icDiv * MATMULPACK;
    int h = mKernelInfo.kernelN;
    param.elh[0] = e;
    param.elh[1] = l;
    param.elh[2] = h;
    param.elhPack[0] = UP_DIV(e, MATMULPACK);
    param.elhPack[1] = UP_DIV(l, MATMULPACK);
    param.elhPack[2] = UP_DIV(h, MATMULPACK);
    param.bStride[0] = 0;
    param.bStride[1] = 1;
    param.bStride[2] = l;

    FuseRegion reg;
    int maxOffsetNumber = 8;
    std::vector<int> offset(maxOffsetNumber);
    auto regionStorage = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(sizeof(FuseRegion));
    auto offsetGpuStorage = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(sizeof(int) * maxOffsetNumber);
    auto offsetGpu = (uint8_t*)offsetGpuStorage.first + offsetGpuStorage.second;

    // Reorder weight
    {
        auto tempCacheBuffer = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(weightSize * sizeof(float));
        float* cacheWeight = (float*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
        runtime->memcpy(cacheWeight, filterDataPtr, weightSize * sizeof(float), MNNMemcpyHostToDevice);
        weightTensor.reset(Tensor::createDevice<int16_t>({param.elhPack[1] * param.elhPack[2] * (MATMULPACK * MATMULPACK)}));
        bn->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
        mFilter = (void *)weightTensor.get()->buffer().device;
        auto& prop = runtime->prop();
        int cores = prop.multiProcessorCount;
        int threadNumbers = prop.maxThreadsPerBlock;
        if (param.elhPack[2] % 2 == 0) {
            KernelReorder<<<cores, threadNumbers>>>((float*)cacheWeight, (half*)mFilter,
                    mKernelInfo.kernelX, mKernelInfo.kernelY, mKernelInfo.kernelC, mKernelInfo.kernelN, 32);
            mUseHPack = true;
        } else {
            KernelReorder<<<cores, threadNumbers>>>((float*)cacheWeight, (half*)mFilter,
                    mKernelInfo.kernelX, mKernelInfo.kernelY, mKernelInfo.kernelC, mKernelInfo.kernelN, MATMULPACK);
        }
        static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempCacheBuffer);
    }

    // Copy Bias
    int biasSize = conv->bias()->size();
    biasTensor.reset(Tensor::createDevice<float>({biasSize}));
    bn->onAcquireBuffer(biasTensor.get(), Backend::STATIC);

    auto tempBiasStorage = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(conv->bias()->size()*sizeof(float));
    auto biasTemp = (float*)((uint8_t*)tempBiasStorage.first + tempBiasStorage.second);
    cuda_check(cudaMemcpy(biasTemp, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));

    // FP32 -> FP16
    mBias = (void *)biasTensor.get()->buffer().device;
    int alignSize = UP_DIV(conv->bias()->size(), PACK_NUMBER) * PACK_NUMBER;
    reg.size[0] = 1;
    reg.size[1] = 1;
    reg.size[2] = alignSize;
    reg.srcStride[0] = 0;
    reg.srcStride[1] = 0;
    reg.srcStride[2] = 1;
    reg.dstStride[0] = 0;
    reg.dstStride[1] = 0;
    reg.dstStride[2] = 1;
    offset[0] = 1;
    offset[1] = 1;
    offset[2] = conv->bias()->size();
    offset[3] = 0;
    offset[4] = 1;
    offset[5] = 1;
    offset[6] = reg.size[2];
    offset[7] = 0;
    reg.fuseNumber = 1;
    runtime->memcpy((uint8_t*)regionStorage.first + regionStorage.second, &reg, sizeof(FuseRegion), MNNMemcpyHostToDevice, true);
    runtime->memcpy(offsetGpu, offset.data(), 8 * sizeof(int), MNNMemcpyHostToDevice, true);
    if (static_cast<CUDABackend*>(bn)->useFp16()) {
        FuseRasterBlitFloatToHalf((uint8_t*)mBias, (uint8_t*)biasTemp, (FuseRegion*)((uint8_t*)regionStorage.first + regionStorage.second), offsetGpu, runtime);
    } else {
        FuseRasterBlitCommon((uint8_t*)mBias, (uint8_t*)biasTemp, (FuseRegion*)((uint8_t*)regionStorage.first + regionStorage.second), offsetGpu, runtime, 4);
    }
    static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(regionStorage);
    static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(offsetGpuStorage);
    static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempBiasStorage);
}

ConvSingleInputExecution::Resource::~Resource() {
    // Do nothing
}
ConvSingleInputExecution::ConvSingleInputExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res) : Execution(backend), mOp(op) {
    mResource = res;
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    auto staticPool = static_cast<CUDABackend*>(backend)->getStaticBufferPool();
    mGpuMatMulParam = staticPool->alloc(sizeof(MatMulParam));
    mGpuIm2ColParam = staticPool->alloc(sizeof(ConvolutionCommon::Im2ColParameter));
}

ConvSingleInputExecution::~ConvSingleInputExecution() {
    auto staticPool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
    staticPool->free(mGpuMatMulParam);
    staticPool->free(mGpuIm2ColParam);
}
bool ConvSingleInputExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvSingleInputExecution(bn, op, mResource);
    *dst = dstExe;
    return true;
}


ErrorCode ConvSingleInputExecution::onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto input = inputs[0], output = outputs[0];
    const int UNIT = PACK_NUMBER;
    auto convCommon = mOp->main_as_Convolution2D()->common();
    auto pads = ConvolutionCommon::convolutionPadFull(input, output, mOp->main_as_Convolution2D()->common());
    int ic = input->channel();
    int icDiv = UP_DIV(ic, PACK_NUMBER);
    mIm2ColParamter.dilateX         = convCommon->dilateX();
    mIm2ColParamter.dilateY         = convCommon->dilateY();
    mIm2ColParamter.strideX         = convCommon->strideX();
    mIm2ColParamter.strideY         = convCommon->strideY();
    mIm2ColParamter.icDiv4          = icDiv;
    mIm2ColParamter.kernelX         = convCommon->kernelX();
    mIm2ColParamter.kernelY         = convCommon->kernelY();
    mIm2ColParamter.padX = std::get<0>(pads);
    mIm2ColParamter.padY = std::get<1>(pads);

    mIm2ColParamter.ih = input->height();
    mIm2ColParamter.iw = input->width();
    mIm2ColParamter.oh = output->height();
    mIm2ColParamter.ow = output->width();
    mIm2ColParamter.srcZStep = input->height() * input->width() * UNIT * input->batch();
    mIm2ColParamter.srcYStep = input->width() * UNIT;
    mIm2ColParamter.packCUnit = UNIT;

    runtime->memcpy((uint8_t*)mGpuIm2ColParam.first + mGpuIm2ColParam.second, &mIm2ColParamter, sizeof(ConvolutionCommon::Im2ColParameter), MNNMemcpyHostToDevice);

    //MNN_PRINT("conv size:%d-%d, %d-%d-%d, %d-%d-%d\n", mIm2ColParamter.kernelX, mIm2ColParamter.strideX, input->height(), input->width(), input->channel(), output->height(), output->width(), output->channel());
    int e = output->height() * output->width() * output->batch();
    int l = icDiv * mIm2ColParamter.kernelX * mIm2ColParamter.kernelY * MATMULPACK;
    int h = output->channel();
    mMatMulParam.elh[0] = e;
    mMatMulParam.elh[1] = l;
    mMatMulParam.elh[2] = h;

    int hPack = MATMULPACK;
    if(mResource->mUseHPack) {
        hPack = 32;
    }
    int ePack = MATMULPACK;
    if(mResource->mUseHPack == false && UP_DIV(e, ePack) % 2 == 0) {
        mUseEPack = true;
        ePack = 32;
    }
    mMatMulParam.elhPack[0] = UP_DIV(e, ePack);
    mMatMulParam.elhPack[1] = UP_DIV(l, MATMULPACK);
    mMatMulParam.elhPack[2] = UP_DIV(h, hPack);
    mMatMulParam.cStride[0] = mIm2ColParamter.ow * mIm2ColParamter.oh * h;
    mMatMulParam.cStride[1] = 1;
    mMatMulParam.cStride[2] = mIm2ColParamter.ow * mIm2ColParamter.oh;
    mMatMulParam.minValue = -FLT_MAX;
    mMatMulParam.maxValue = FLT_MAX;
    if (convCommon->relu()) {
        mMatMulParam.minValue = 0.0f;
    }
    if (convCommon->relu6()) {
        mMatMulParam.minValue = 0.0f;
        mMatMulParam.maxValue = 6.0f;
    }
    //MNN_PRINT("Im2Col：%d-%d-%d temp size:%zu!!!\n\n",output->width(), ic, mIm2ColParamter.kernelX, (size_t)sizeof(__half) * mMatMulParam.elhPack[0] * mMatMulParam.elhPack[1] * MATMULPACK * MATMULPACK);
    // When Im2Col memory size big than 2GB
    if((size_t)mMatMulParam.elhPack[0] * (size_t)ePack * (size_t)mMatMulParam.elhPack[1] * (size_t)hPack > 1024*1024*1024 && mIm2ColParamter.kernelX > 1 && mIm2ColParamter.kernelY > 1) {
        //printf("need im2col in block\n");
        mIsBlock = true;
        mBlockNum = 16;
        mMatMulParam.elhPack[0] = UP_DIV(mMatMulParam.elhPack[0], mBlockNum);
    }
    runtime->memcpy((uint8_t*)mGpuMatMulParam.first + mGpuMatMulParam.second, &mMatMulParam, sizeof(MatMulParam), MNNMemcpyHostToDevice);

    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    auto buffer = pool->alloc((size_t)sizeof(__half) * (size_t)mMatMulParam.elhPack[0] * (size_t)mMatMulParam.elhPack[1] * (size_t)ePack * (size_t)hPack);
    mIm2ColBuffer = (__half*)((uint8_t*)buffer.first + buffer.second);
    pool->free(buffer);
    
    return NO_ERROR;
}

ErrorCode ConvSingleInputExecution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    //MNN_PRINT("cuda convSingleInput onExecute in, inputsize:%d %d\n", (int)inputs.size(), workspace_size_);
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(input);
    const void *input_addr = (const void*)inputs[0]->deviceId();
    const void *filter_addr = mResource->mFilter;
    const void *bias_addr = mResource->mBias;
    auto bn = backend();
    void *output_addr = (void*)outputs[0]->deviceId();

    auto gpuIm2Col = (const ConvolutionCommon::Im2ColParameter*)((uint8_t*)mGpuIm2ColParam.first + mGpuIm2ColParam.second);
    auto gpuMatMul = (const MatMulParam*)((uint8_t*)mGpuMatMulParam.first + mGpuMatMulParam.second);

    // Im2col in Block
    for(int block_idx = 0; block_idx < mBlockNum; block_idx++) {
        if(mUseEPack) {
            Im2ColMain(runtime, &mMatMulParam, gpuMatMul, &mIm2ColParamter, gpuIm2Col, (const Tensor*)(inputs[0]), mIm2ColBuffer, 32, 5, bytes, block_idx);
            GemmPacked32x16(runtime, &mMatMulParam, gpuMatMul, (float*)output_addr, (const __half*)mIm2ColBuffer, (const __half*)filter_addr, (const half*)bias_addr, bytes, block_idx);
        } else {
            Im2ColMain(runtime, &mMatMulParam, gpuMatMul, &mIm2ColParamter, gpuIm2Col, (const Tensor*)(inputs[0]), mIm2ColBuffer, 16, 4, bytes, block_idx);
            if (mResource->mUseHPack) {
                GemmPacked16x32(runtime, &mMatMulParam, gpuMatMul, (float*)output_addr, (const __half*)mIm2ColBuffer, (const __half*)filter_addr, (const half*)bias_addr, bytes, block_idx);
            } else {
                //printf("NotPack:%d-%d-%d-%d-%d, %d-%d-%d\n", mIm2ColParamter.icDiv4, mIm2ColParamter.ih, mIm2ColParamter.iw, mIm2ColParamter.oh, mIm2ColParamter.ow, mMatMulParam.elhPack[0], mMatMulParam.elhPack[1], mMatMulParam.elhPack[2]);
                GemmPackedFullMain(runtime, &mMatMulParam, gpuMatMul, (float*)output_addr, (const __half*)mIm2ColBuffer, (const __half*)filter_addr, (const half*)bias_addr, bytes, block_idx);
            }
        }
    }
    return NO_ERROR;
}

class CUDAConvolutionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, 
            const MNN::Op* op, Backend* backend) const override {
        if (nullptr != op->main_as_Convolution2D()->quanParameter()) {
            auto quan = op->main_as_Convolution2D()->quanParameter();
            if (1 == quan->type() || 2 == quan->type()) {
                if (quan->has_scaleInt()) {
                    // Don't support IDST-int8 because of error
                    return nullptr;
                }
            }
        }

        auto conv = op->main_as_Convolution2D()->common();
        if(ConvWinogradExecution::isValid(op->main_as_Convolution2D(), inputs[0])) {
            //printf("%dx%ds%dd%d\n", conv->kernelX(), conv->kernelY(), conv->strideX(), conv->dilateX());

            std::shared_ptr<ConvWinogradExecution::Resource> resource(new ConvWinogradExecution::Resource(backend, op));
            return new ConvWinogradExecution(backend, op, resource);
        }

        // std::shared_ptr<ConvSingleInputExecution::Resource> resource(new ConvSingleInputExecution::Resource(backend, op));
        // return new ConvSingleInputExecution(backend, op, resource);

        std::shared_ptr<ConvCutlassExecution::Resource> resource(new ConvCutlassExecution::Resource(backend, op));
        return new ConvCutlassExecution(backend, op, resource);
    }
};

CUDACreatorRegister<CUDAConvolutionCreator> __ConvExecution(OpType_Convolution);

}// namespace CUDA
}// namespace MNN
