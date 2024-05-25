//
//  MultiInputConvExecution.cpp
//  MNN
//
//  Created by MNN on 2023/03/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MultiInputConvExecution.hpp"
#include "Raster.cuh"
#include "ConvBaseKernel.cuh"

//#define DEBUG

namespace MNN {
namespace CUDA {

MultiInputConvExecution::MultiInputConvExecution(const MNN::Op* op, Backend* backend) : CutlassConvCommonExecution(backend) {
    mOp = op;
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    mPrecisonLevel = static_cast<CUDABackend*>(backend)->getPrecision();
    mFp16Infer = (mPrecisonLevel == 2);
    mFp32Infer = (mPrecisonLevel == 1);
    mFp16Fp32MixInfer = (mPrecisonLevel == 0);
}

MultiInputConvExecution::~MultiInputConvExecution() {

}


ErrorCode MultiInputConvExecution::onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();

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


    mNeedWeightFill = ((mGemmInfo.elh[1] != mGemmInfo.elhPad[1]) || (mGemmInfo.elh[2] != mGemmInfo.elhPad[2]));
    mNeedBiasFill   = (inputs.size() > 2) && (mGemmInfo.elh[2] != mGemmInfo.elhPad[2]);
    // Reorder weight

    size_t elementBytes = 2;
    // Only when fp32 Im2Col convert to fp32, Fp16Fp32Mix Im2Col convert to fp16
    if(mFp32Infer) {
        elementBytes = 4;
    }

    MemChunk bufferFilter;
    if(mNeedWeightFill) {
        bufferFilter = pool->alloc(elementBytes * (size_t)mGemmInfo.elhPad[1] * (size_t)mGemmInfo.elhPad[2]);
        mFilterAddr = (void*)(bufferFilter.ptr());
    } else {
        mFilterAddr = (void*)inputs[1]->deviceId();
    }

    // Copy Bias
    MemChunk bufferBias;
    if(mNeedBiasFill) {
        bufferBias = pool->alloc(elementBytes * (size_t)mGemmInfo.elhPad[2]);
        mBiasAddr = (void*)(bufferBias.ptr());

    } else {
        mBiasAddr = (void*)inputs[2]->deviceId();
    }


    mIsConv1x1S1D1P0 = (mIm2ColParamter.kernelX == 1 && mIm2ColParamter.kernelY == 1 && \
                        mIm2ColParamter.strideX == 1 && mIm2ColParamter.strideY == 1 && \
                        mIm2ColParamter.dilateX == 1 && mIm2ColParamter.dilateY == 1 && \
                        mIm2ColParamter.padX == 0 && mIm2ColParamter.padY == 0);
    mNeedIm2Col = !(mIsConv1x1S1D1P0 && (mFp16Infer || mFp32Infer));

    MemChunk bufferIm2Col;
    if(mNeedIm2Col) {
        bufferIm2Col = pool->alloc(elementBytes * (size_t)mGemmInfo.elh[0] * (size_t)mGemmInfo.elhPad[1]);
        mIm2ColBuffer = (void*)(bufferIm2Col.ptr());
    }

    // free for Reuse
    if(mNeedWeightFill) {
        pool->free(bufferFilter);
    }
    if(mNeedBiasFill) {
        pool->free(bufferBias);
    }
    if(mNeedIm2Col) {
        pool->free(bufferIm2Col);
    }

    // Call from different function
    if(mFp32Infer){
        return callCutlassGemmCudaCoreFloat32(inputs, outputs);
    } 

    mGpuComputeCap = runtime->compute_capability();
    //MNN_PRINT("Gpu smArch is sm_%d\n", mGpuComputeCap);
    if(mGpuComputeCap < 70) {
        return callCutlassGemmCudaCoreFloat16(inputs, outputs);
    } else if(mGpuComputeCap < 75) {
        return callCutlassGemmTensorCore884(inputs, outputs);
    }

    return callCutlassGemmTensorCore(inputs, outputs);
}

ErrorCode MultiInputConvExecution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];

    //MNN_PRINT("cutlass hw:%d-%d\n", input->height(), input->width());
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    const void *input_addr = (const void*)inputs[0]->deviceId();
    auto bn = backend();
    void *output_addr = (void*)outputs[0]->deviceId();

    // Im2col in Block
    for(int block_idx = 0; block_idx < mBlockNum; block_idx++) {
        if(mIsConv1x1S1D1P0 && mFp16Fp32MixInfer) {
            size_t maxCount = mGemmInfo.elh[0] * mGemmInfo.elhPad[1];
            callFloat2Half((const void*)input_addr, (void*)mIm2ColBuffer, maxCount, runtime);

        } else if (mNeedIm2Col) {
            callIm2ColPack((const void *)input_addr, (void *)mIm2ColBuffer, &mIm2ColParamter, mGemmInfo.elh[0], mGemmInfo.elh[1], mGemmInfo.elhPad[0], mGemmInfo.elhPad[1], mPrecisonLevel, runtime);
        }
    }

    if(mNeedWeightFill) {
        callWeightFill((const void *)inputs[1]->deviceId(), (void *)mFilterAddr, mIm2ColParamter.ic, mGemmInfo.elh[1], mGemmInfo.elh[2], mGemmInfo.elhPad[1], mGemmInfo.elhPad[2], mPrecisonLevel, runtime);
    }

    if(mNeedBiasFill) {
        if(mFp16Fp32MixInfer) {
            runtime->memset(mBiasAddr, 0, mGemmInfo.elhPad[2] * sizeof(int16_t));

            callFloat2Half((const void*)inputs[2]->deviceId(), (void*)mBiasAddr, mGemmInfo.elhPad[2], runtime);
        } else {
            if(mFp32Infer) {
                runtime->memset(mBiasAddr, 0, mGemmInfo.elhPad[2] * sizeof(int32_t));
                runtime->memcpy(mBiasAddr, (const void *)inputs[2]->deviceId(), mGemmInfo.elh[2] * sizeof(int32_t), MNNMemcpyDeviceToDevice);
            } else {
                runtime->memset(mBiasAddr, 0, mGemmInfo.elhPad[2] * sizeof(int16_t));
                runtime->memcpy(mBiasAddr, (const void *)inputs[2]->deviceId(), mGemmInfo.elh[2] * sizeof(int16_t), MNNMemcpyDeviceToDevice);
            }

        }
    }

    // Run cutlass gemm forward
    return runCutlassGemmFunc();
}


}// namespace CUDA
}// namespace MNN