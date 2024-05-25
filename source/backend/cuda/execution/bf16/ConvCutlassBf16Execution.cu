//
//  ConvCutlassBf16Execution.cpp
//  MNN
//
//  Created by MNN on 2023/05/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef ENABLE_CUDA_BF16

#include "ConvCutlassBf16Execution.hpp"
#include "../ConvBaseKernel.cuh"

//#define DEBUG

namespace MNN {
namespace CUDA {

ConvCutlassBf16Execution::Resource::Resource(Backend* bn, const MNN::Op* op) {
    mBackend = bn;
    auto runtime = static_cast<CUDABackend*>(bn)->getCUDARuntime();

    auto conv       = op->main_as_Convolution2D();
    auto common     = conv->common();

    //weight host->device
    const float* filterDataPtr = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, bn, conv, &filterDataPtr, &weightSize);
    auto oc = common->outputCount();

    int l = weightSize / oc;
    int h = oc;

    int ic = common->inputCount();
    if(ic == 0) {
        ic = l / common->kernelX() / common->kernelY();
    }

    int lp = UP_DIV(l, 8) * 8;
    int hp = UP_DIV(h, 8) * 8;

    // Reorder weight
    {
        auto tempCacheBuffer = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(weightSize * sizeof(float));
        float* cacheWeight = (float*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
        runtime->memcpy(cacheWeight, filterDataPtr, weightSize * sizeof(float), MNNMemcpyHostToDevice);
        weightTensor.reset(Tensor::createDevice<int16_t>({lp * hp}));
        bn->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
        mFilter = (void *)weightTensor.get()->buffer().device;

        // From Float32 To Bfloat16
        callWeightFill((const void *)cacheWeight, (void *)mFilter, ic, l, h, lp, hp, 3, runtime);

        static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempCacheBuffer);
    }

    // Copy Bias
    {
        int biasSize = conv->bias()->size();
        int hp = UP_DIV(biasSize, 8) * 8;

        auto tempBiasStorage = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(hp*sizeof(float));
        auto biasTemp = (float*)((uint8_t*)tempBiasStorage.first + tempBiasStorage.second);
        runtime->memset(biasTemp, 0, hp * sizeof(int32_t));
        cuda_check(cudaMemcpy(biasTemp, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));

        biasTensor.reset(Tensor::createDevice<int16_t>({hp}));
        bn->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
        mBias = (void *)biasTensor.get()->buffer().device;
        callFloat2BFloat16((const void*)biasTemp, (void*)mBias, hp, runtime);

        static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempBiasStorage);
    }
}

ConvCutlassBf16Execution::Resource::~Resource() {
    // Do nothing
}
ConvCutlassBf16Execution::ConvCutlassBf16Execution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res) : CutlassConvCommonExecution(backend) {
    mOp = op;
    mResource = res;
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    mPrecisonLevel = static_cast<CUDABackend*>(backend)->getPrecision();
    MNN_ASSERT(mPrecisonLevel == 3);
    mBf16Infer = true;
}

ConvCutlassBf16Execution::~ConvCutlassBf16Execution() {

}
bool ConvCutlassBf16Execution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvCutlassBf16Execution(bn, op, mResource);
    *dst = dstExe;
    return true;
}


ErrorCode ConvCutlassBf16Execution::onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto input = inputs[0], output = outputs[0];
    const int UNIT = PACK_NUMBER;
    auto convCommon = mOp->main_as_Convolution2D()->common();
    auto pads = ConvolutionCommon::convolutionPadFull(input, output, mOp->main_as_Convolution2D()->common());
    int ic = input->channel();
    auto icDiv = UP_DIV(ic, UNIT);

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
    mIm2ColParamter.ic = ic;
    mIm2ColParamter.oh = output->height();
    mIm2ColParamter.ow = output->width();
    mIm2ColParamter.srcZStep = input->height() * input->width() * UNIT * input->batch();
    mIm2ColParamter.srcYStep = input->width() * UNIT;
    mIm2ColParamter.packCUnit = UNIT;

    mActivationType = convCommon->relu() ? 1 : convCommon->relu6() ? 2 : 0;

    //MNN_PRINT("conv size:%d-%d, %d-%d-%d, %d-%d-%d\n", mIm2ColParamter.kernelX, mIm2ColParamter.strideX, input->height(), input->width(), input->channel(), output->height(), output->width(), output->channel());
    int e = output->height() * output->width() * output->batch();
    int l = ic * mIm2ColParamter.kernelX * mIm2ColParamter.kernelY;
    int h = output->channel();
    mGemmInfo.elh[0] = e;
    mGemmInfo.elh[1] = l;
    mGemmInfo.elh[2] = h;
    mGemmInfo.elhPad[0] = UP_DIV(e, 8) * 8;
    mGemmInfo.elhPad[1] = UP_DIV(l, 8) * 8;
    mGemmInfo.elhPad[2] = UP_DIV(h, 8) * 8;

    //MNN_PRINT("Activate:%d \n", mActivationType);
    //MNN_PRINT("Im2Col：%d-%d-%d temp size:%zu!!!\n\n",output->width(), ic, mIm2ColParamter.kernelX, (size_t)sizeof(__half) * mMatMulParam.elhPack[0] * mMatMulParam.elhPack[1] * MATMULPACK * MATMULPACK);
    // When Im2Col memory size big than 2GB
    if(0){//(size_t)mGemmInfo.elh[0] * (size_t)mGemmInfo.elh[1] > 1024*1024*1024 && mIm2ColParamter.kernelX > 1 && mIm2ColParamter.kernelY > 1) {
        //printf("need im2col in block\n");
        mIsBlock = true;
        mBlockNum = 16;
        mGemmInfo.elh[0] = UP_DIV(mGemmInfo.elh[0], mBlockNum);
    }

    mIsConv1x1S1D1P0 = (mIm2ColParamter.kernelX == 1 && mIm2ColParamter.kernelY == 1 && \
                        mIm2ColParamter.strideX == 1 && mIm2ColParamter.strideY == 1 && \
                        mIm2ColParamter.dilateX == 1 && mIm2ColParamter.dilateY == 1 && \
                        mIm2ColParamter.padX == 0 && mIm2ColParamter.padY == 0);
    mNeedIm2Col = !(mIsConv1x1S1D1P0);

    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    if(mNeedIm2Col) {
        size_t im2colBytes = 2;
        // Only when fp32 Im2Col convert to fp32, Fp16Fp32Mix Im2Col convert to fp16
        if(mFp32Infer) {
            im2colBytes = 4;
        }
        auto buffer = pool->alloc(im2colBytes * (size_t)mGemmInfo.elh[0] * (size_t)mGemmInfo.elhPad[1]);
        mIm2ColBuffer = (void*)((uint8_t*)buffer.first + buffer.second);
        pool->free(buffer);
    }


    mFilterAddr = mResource->mFilter;
    mBiasAddr   = mResource->mBias;
    mBackendPtr = mResource->mBackend;

    //MNN_PRINT("Gpu smArch is sm_%d\n", mGpuComputeCap);
    return callCutlassGemmBf16TensorCore(inputs, outputs);
}

ErrorCode ConvCutlassBf16Execution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    //MNN_PRINT("cuda convSingleInput onExecute in, inputsize:%d %d\n", (int)inputs.size(), workspace_size_);
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];

    //printf("convcutlass:%p %p\n", input->deviceId(), output->deviceId());
    //MNN_PRINT("cutlass hw:%d-%d\n", input->height(), input->width());
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    const void *input_addr = (const void*)inputs[0]->deviceId();
    const void *filter_addr = mResource->mFilter;
    const void *bias_addr = mResource->mBias;
    auto bn = backend();
    void *output_addr = (void*)outputs[0]->deviceId();

    const int sw = mIm2ColParamter.strideX;
    const int sh = mIm2ColParamter.strideY;
    const int dw = mIm2ColParamter.dilateX;
    const int dh = mIm2ColParamter.dilateY;
    const int pw = mIm2ColParamter.padX;
    const int ph = mIm2ColParamter.padY;
    const int icDiv4 = mIm2ColParamter.icDiv4;
    const int iw = mIm2ColParamter.iw;
    const int ih = mIm2ColParamter.ih;

    //printf("%d-%d-%d-%d-%d, %d-%d\n", cpuIm2Col->icDiv4, cpuIm2Col->ih, cpuIm2Col->iw, cpuIm2Col->oh, cpuIm2Col->ow, eAlign, lAlign);
    // Im2col in Block
    for(int block_idx = 0; block_idx < mBlockNum; block_idx++) {
        if (mNeedIm2Col) {
            callIm2ColPack((const void *)input_addr, (void *)mIm2ColBuffer, &mIm2ColParamter, mGemmInfo.elh[0], mGemmInfo.elh[1], \
                mGemmInfo.elhPad[0], mGemmInfo.elhPad[1], mPrecisonLevel, runtime);
        }
    }

    // Run cutlass gemm forward
    return runCutlassGemmFunc();
}


}// namespace CUDA
}// namespace MNN
#endif