//
//  MultiInputDeconvExecution.cpp
//  MNN
//
//  Created by MNN on 2023/04/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MultiInputDeconvExecution.hpp"
#include "ConvBaseKernel.cuh"
#include "DeconvBaseKernel.cuh"

//#define DEBUG

namespace MNN {
namespace CUDA {

MultiInputDeconvExecution::MultiInputDeconvExecution(const MNN::Op* op, Backend* backend) : CutlassDeconvCommonExecution(backend) {
    mOp = op;
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    mPrecisonLevel = static_cast<CUDABackend*>(backend)->getPrecision();
    mFp16Infer = (mPrecisonLevel == 2);
    mFp32Infer = (mPrecisonLevel == 1);
    mFp16Fp32MixInfer = (mPrecisonLevel == 0);
}

MultiInputDeconvExecution::~MultiInputDeconvExecution() {

}


ErrorCode MultiInputDeconvExecution::onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto input = inputs[0], output = outputs[0];
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]);
    auto convCommon = mOp->main_as_Convolution2D()->common();

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


    mActivationType = convCommon->relu() ? 1 : convCommon->relu6() ? 2 : 0;


    mKernelInfo.kernelX        = convCommon->kernelX();
    mKernelInfo.kernelY        = convCommon->kernelY();
    mKernelInfo.groups         = convCommon->group();
    mKernelInfo.strideX        = convCommon->strideX();
    mKernelInfo.strideY        = convCommon->strideY();
    mKernelInfo.dilateX        = convCommon->dilateX();
    mKernelInfo.dilateY        = convCommon->dilateY();
    mKernelInfo.activationType = mActivationType;
    mKernelInfo.kernelN        = output->channel();
    mKernelInfo.kernelC        = input->channel();

    // Matmul Param
    int e = output->channel() * mKernelInfo.kernelX * mKernelInfo.kernelY;
    int l = input->channel();
    int h = input->height() * input->width() * output->batch();

    mGemmInfo.elh[0] = e;
    mGemmInfo.elh[1] = l;
    mGemmInfo.elh[2] = h;
    mGemmInfo.elhPad[0] = UP_DIV(e, PACK_NUMBER) * PACK_NUMBER;
    mGemmInfo.elhPad[1] = UP_DIV(l, PACK_NUMBER) * PACK_NUMBER;
    mGemmInfo.elhPad[2] = UP_DIV(h, PACK_NUMBER) * PACK_NUMBER;


    // Alloc temp cuda memory
    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    MemChunk buffer_input, buffer_im2col;
    if(mFp16Fp32MixInfer) {
        buffer_input = pool->alloc(sizeof(__half) * mGemmInfo.elhPad[1] * mGemmInfo.elh[2]);
        mInputBuffer = (void*)buffer_input.ptr();
    } else {
        mInputBuffer = (void*)input->deviceId();
    }
    buffer_im2col = pool->alloc(bytes * mGemmInfo.elh[0] * mGemmInfo.elhPad[2]);
    mIm2ColBuffer = (void*)buffer_im2col.ptr();

    mNeedWeightFill = (mGemmInfo.elh[1] != mGemmInfo.elhPad[1]);
    MemChunk buffer_filter;
    if(mNeedWeightFill) {
        buffer_filter = pool->alloc(bytes * (size_t)mGemmInfo.elh[0] * (size_t)mGemmInfo.elhPad[1]);
        mFilterAddr = (void*)buffer_filter.ptr();
    } else {
        mFilterAddr = (void*)inputs[1]->deviceId();
    }


    if(mFp16Fp32MixInfer || mFp32Infer) {
        mZeroTensor.reset(Tensor::createDevice<uint32_t>({mGemmInfo.elhPad[2]}));
    } else {
        mZeroTensor.reset(Tensor::createDevice<uint16_t>({mGemmInfo.elhPad[2]}));
    }
    static_cast<CUDABackend*>(backend())->onAcquireBuffer(mZeroTensor.get(), Backend::STATIC);

    mZeroPtr = (void *)mZeroTensor.get()->buffer().device;
    cuda_check(cudaMemset(mZeroPtr, 0, mGemmInfo.elhPad[2]*bytes));


    // free for Reuse
    if(mFp16Fp32MixInfer) {
        pool->free(buffer_input);
    }
    pool->free(buffer_im2col);
    if(mNeedWeightFill) {
        pool->free(buffer_filter);
    }
 
    // Call from different function
    if(mFp32Infer){
        return callCutlassGemmCudaCoreFloat32(inputs, outputs);
    } 
 
    mGpuComputeCap = runtime->compute_capability();
    //MNN_PRINT("Gpu smArch is sm_%d\n", mGpuComputeCap);
    if(mGpuComputeCap < 75) {
        return callCutlassGemmCudaCoreFloat16(inputs, outputs);
    }
    return callCutlassGemmTensorCore(inputs, outputs);
}

ErrorCode MultiInputDeconvExecution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    const void *input_addr = (const void*)inputs[0]->deviceId();
    void *output_addr = (void*)outputs[0]->deviceId();
    if(inputs.size() > 2) {
        mBiasAddr = (void*)inputs[2]->deviceId();
    }
    
    // Do input convert
    if(mFp16Fp32MixInfer) {
        size_t maxCount = mGemmInfo.elhPad[1] * mGemmInfo.elh[2];
        callFloat2Half((const void*)input_addr, (void*)mInputBuffer, maxCount, runtime);
    }

    // Do weight Reoreder
    if(mNeedWeightFill) {
        callWeightReorder((const void *)inputs[1]->deviceId(), (void *)mFilterAddr, mKernelInfo, mGemmInfo.elhPad[1], mPrecisonLevel, runtime);
    }

    // Run cutlass gemm forward
    runCutlassGemmFunc();

    // Run Col2Im
    int convert_flag = mPrecisonLevel;
    if(convert_flag == 0) {
        convert_flag = 1;
    }
    callCol2ImFunc((const void*)mIm2ColBuffer, (const void*)mBiasAddr, (void *)output_addr, &mCol2ImParamter, convert_flag, runtime);
    
    return NO_ERROR;
}


}// namespace CUDA
}// namespace MNN