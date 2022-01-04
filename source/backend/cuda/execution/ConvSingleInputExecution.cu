//
//  ConvSingleInputExecution.cpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvSingleInputExecution.hpp"

namespace MNN {
namespace CUDA {

__global__ void Im2Col(const ConvolutionCommon::Im2ColParameter* param,
        const MatMulParam* matmulParam,
        const float* A,
        __half* AP) {
    int eAlign = matmulParam->elhPack[0] * MATMULPACK;
    int lAlign = matmulParam->elhPack[1] * MATMULPACK;
    int maxCount = eAlign * lAlign;
    int kernelCount = param->kernelX * param->kernelY;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int eIndex = index % eAlign;
        int lIndex = index / eAlign;
        // Compute for dest
        int eU = eIndex / MATMULPACK;
        int eR = eIndex % MATMULPACK;
        int lU = lIndex / MATMULPACK;
        int lR = lIndex % MATMULPACK;
        auto dstOffset = eU * matmulParam->elhPack[1] * (MATMULPACK * MATMULPACK) + lU * (MATMULPACK * MATMULPACK) + eR * MATMULPACK + lR;
        if (eIndex >= matmulParam->elh[0] || lIndex >= matmulParam->elh[1]) {
            AP[dstOffset] = 0.0;
            continue;
        }
        // Compute for source
        int ox = eIndex % param->ow;
        int oy = eIndex / param->ow;
        int ob = oy / param->oh;
        oy = oy % param->oh;
        int sz = lIndex / kernelCount;
        int kI = lIndex % kernelCount;
        int ksx = kI % param->kernelX;
        int ksy = kI / param->kernelX;

        int sx = ox * param->strideX + ksx * param->dilateX - param->padX;
        int sy = oy * param->strideY + ksy * param->dilateY - param->padY;
        if (sx >= 0 && sx < param->iw) {
            if (sy >=0 && sy < param->ih) {
                __half value = A[sz * param->ih * param->iw + ob * param->iw * param->ih * param->icDiv4 + sy * param->iw + sx];
                AP[dstOffset] = value;
                continue;
            }
        }
        AP[dstOffset] = 0.0;
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

    MatMulParam param;
    int e = 0;
    int l = mKernelInfo.kernelX * mKernelInfo.kernelY * mKernelInfo.kernelC;
    int h = mKernelInfo.kernelN;
    param.elh[0] = e;
    param.elh[1] = l;
    param.elh[2] = h;
    param.elhPack[0] = UP_DIV(e, 16);
    param.elhPack[1] = UP_DIV(l, 16);
    param.elhPack[2] = UP_DIV(h, 16);
    param.bStride[0] = 0;
    param.bStride[1] = 1;
    param.bStride[2] = l;

    auto gpuParam = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(sizeof(MatMulParam));
    auto tempCacheBuffer = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(weightSize * sizeof(float));
    float* cacheWeight = (float*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
    runtime->memcpy(cacheWeight, filterDataPtr, weightSize * sizeof(float), MNNMemcpyHostToDevice);
    runtime->memcpy((uint8_t*)gpuParam.first + gpuParam.second, &param, sizeof(MatMulParam), MNNMemcpyHostToDevice);
    // Reorder weight
    weightTensor.reset(Tensor::createDevice<int16_t>({param.elhPack[1] * param.elhPack[2] * (MATMULPACK * MATMULPACK)}));
    bn->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
    mFilter = (void *)weightTensor.get()->buffer().device;
    GemmPrepareRerange(runtime, &param, (const MatMulParam*)((uint8_t*)gpuParam.first + gpuParam.second), nullptr, nullptr, cacheWeight, (__half*)mFilter);
    static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempCacheBuffer);
    static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(gpuParam);

    // Copy Bias
    int biasSize = conv->bias()->size();
    biasTensor.reset(Tensor::createDevice<float>({biasSize}));
    bn->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
    mBias = (void *)biasTensor.get()->buffer().device;
    cuda_check(cudaMemcpy(mBias, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));
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
    const int UNIT = 1;
    auto convCommon = mOp->main_as_Convolution2D()->common();
    auto pads = ConvolutionCommon::convolutionPadFull(input, output, mOp->main_as_Convolution2D()->common());
    mIm2ColParamter.dilateX         = convCommon->dilateX();
    mIm2ColParamter.dilateY         = convCommon->dilateY();
    mIm2ColParamter.strideX         = convCommon->strideX();
    mIm2ColParamter.strideY         = convCommon->strideY();
    mIm2ColParamter.icDiv4          = input->channel();
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

    int e = output->height() * output->width() * output->batch();
    int l = input->channel() * mIm2ColParamter.kernelX * mIm2ColParamter.kernelY;
    int h = output->channel();
    mMatMulParam.elh[0] = e;
    mMatMulParam.elh[1] = l;
    mMatMulParam.elh[2] = h;
    mMatMulParam.elhPack[0] = UP_DIV(e, 16);
    mMatMulParam.elhPack[1] = UP_DIV(l, 16);
    mMatMulParam.elhPack[2] = UP_DIV(h, 16);
    mMatMulParam.cStride[0] = mIm2ColParamter.ow * mIm2ColParamter.oh * h;
    mMatMulParam.cStride[1] = 1;
    mMatMulParam.cStride[2] = mIm2ColParamter.ow * mIm2ColParamter.oh;
    mMatMulParam.split[0] = 1;
    mMatMulParam.split[1] = 1;
    mMatMulParam.split[2] = mIm2ColParamter.ow * mIm2ColParamter.oh;
    if (convCommon->relu()) {
        mMatMulParam.minValue = 0.0f;
    }
    if (convCommon->relu6()) {
        mMatMulParam.minValue = 0.0f;
        mMatMulParam.maxValue = 6.0f;
    }
    runtime->memcpy((uint8_t*)mGpuMatMulParam.first + mGpuMatMulParam.second, &mMatMulParam, sizeof(MatMulParam), MNNMemcpyHostToDevice);

    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    auto buffer = pool->alloc(sizeof(__half) * mMatMulParam.elhPack[0] * mMatMulParam.elhPack[1] * MATMULPACK * MATMULPACK);
    mIm2ColBuffer = (__half*)((uint8_t*)buffer.first + buffer.second);
    pool->free(buffer);
    return NO_ERROR;
}

ErrorCode ConvSingleInputExecution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    //MNN_PRINT("cuda convSingleInput onExecute in, inputsize:%d %d\n", (int)inputs.size(), workspace_size_);
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    const void *input_addr = (const void*)inputs[0]->deviceId();
    const void *filter_addr = mResource->mFilter;
    const void *bias_addr = mResource->mBias;

    void *output_addr = (void*)outputs[0]->deviceId();
    auto& prop = runtime->prop();
    int threads_num = prop.maxThreadsPerBlock;
    int cores = prop.multiProcessorCount;
    auto gpuIm2Col = (const ConvolutionCommon::Im2ColParameter*)((uint8_t*)mGpuIm2ColParam.first + mGpuIm2ColParam.second);
    auto gpuMatMul = (const MatMulParam*)((uint8_t*)mGpuMatMulParam.first + mGpuMatMulParam.second);
    //runtime->memset(mIm2ColBuffer, 0, mMatMulParam.elhPack[0] * mMatMulParam.elhPack[1] * sizeof(__half) * (MATMULPACK * MATMULPACK));
    Im2Col<<<cores, threads_num>>>(gpuIm2Col, gpuMatMul, (const float*)input_addr, mIm2ColBuffer);
    GemmPackedMain(runtime, &mMatMulParam, gpuMatMul, (float*)output_addr, (const __half*)mIm2ColBuffer, (const __half*)filter_addr, (const float*)bias_addr);

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
        std::shared_ptr<ConvSingleInputExecution::Resource> resource(new ConvSingleInputExecution::Resource(backend, op));
        return new ConvSingleInputExecution(backend, op, resource);
    }
};

CUDACreatorRegister<CUDAConvolutionCreator> __ConvExecution(OpType_Convolution);

}// namespace CUDA
}// namespace MNN
