//
//  PoolExecution.cu
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "PoolExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/musa/core/MusaBackend.hpp"
#include <musa_runtime.h>

namespace MNN {
namespace MUSA {

// MUSA kernel for max pooling
__global__ void MaxPoolKernel(const float* input, float* output, 
                              int batch, int channels, 
                              int inputHeight, int inputWidth,
                              int outputHeight, int outputWidth,
                              int kernelHeight, int kernelWidth,
                              int strideHeight, int strideWidth,
                              int padHeight, int padWidth) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batch * channels * outputHeight * outputWidth;
    
    if (index >= totalSize) return;
    
    int tmp = index;
    int outW = tmp % outputWidth;
    tmp /= outputWidth;
    int outH = tmp % outputHeight;
    tmp /= outputHeight;
    int channel = tmp % channels;
    int batchIdx = tmp / channels;
    
    int inWOrigin = outW * strideWidth - padWidth;
    int inHOrigin = outH * strideHeight - padHeight;
    
    float maxVal = -FLT_MAX;
    
    for (int kh = 0; kh < kernelHeight; kh++) {
        for (int kw = 0; kw < kernelWidth; kw++) {
            int inW = inWOrigin + kw;
            int inH = inHOrigin + kh;
            
            if (inH >= 0 && inH < inputHeight && inW >= 0 && inW < inputWidth) {
                int inputIndex = ((batchIdx * channels + channel) * inputHeight + inH) * inputWidth + inW;
                float val = input[inputIndex];
                if (val > maxVal) {
                    maxVal = val;
                }
            }
        }
    }
    
    output[index] = maxVal;
}

// MUSA kernel for average pooling
__global__ void AvgPoolKernel(const float* input, float* output, 
                              int batch, int channels, 
                              int inputHeight, int inputWidth,
                              int outputHeight, int outputWidth,
                              int kernelHeight, int kernelWidth,
                              int strideHeight, int strideWidth,
                              int padHeight, int padWidth) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batch * channels * outputHeight * outputWidth;
    
    if (index >= totalSize) return;
    
    int tmp = index;
    int outW = tmp % outputWidth;
    tmp /= outputWidth;
    int outH = tmp % outputHeight;
    tmp /= outputHeight;
    int channel = tmp % channels;
    int batchIdx = tmp / channels;
    
    int inWOrigin = outW * strideWidth - padWidth;
    int inHOrigin = outH * strideHeight - padHeight;
    
    float sum = 0.0f;
    int count = 0;
    
    for (int kh = 0; kh < kernelHeight; kh++) {
        for (int kw = 0; kw < kernelWidth; kw++) {
            int inW = inWOrigin + kw;
            int inH = inHOrigin + kh;
            
            if (inH >= 0 && inH < inputHeight && inW >= 0 && inW < inputWidth) {
                int inputIndex = ((batchIdx * channels + channel) * inputHeight + inH) * inputWidth + inW;
                sum += input[inputIndex];
                count++;
            }
        }
    }
    
    output[index] = sum / count;
}

PoolExecution::PoolExecution(PoolType type, const std::vector<int>& kernels, const std::vector<int>& strides, 
                             const std::vector<int>& pads, Backend* backend) : Execution(backend) {
    auto musaBackend = static_cast<MusaBackend*>(backend);
    mRuntime = musaBackend->getMusaRuntime();
    mType = type;
    mKernels = kernels;
    mStrides = strides;
    mPads = pads;
}

ErrorCode PoolExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto shape = input->shape();
    
    mBatch = shape[0];
    mChannels = shape[1];
    mInputHeight = shape[2];
    mInputWidth = shape[3];
    
    auto output = outputs[0];
    auto outputShape = output->shape();
    mOutputHeight = outputShape[2];
    mOutputWidth = outputShape[3];
    
    return NO_ERROR;
}

ErrorCode PoolExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start PoolExecution onExecute...\n");
#endif
    
    auto input = inputs[0]->deviceId();
    auto output = outputs[0]->deviceId();
    
    int totalSize = mBatch * mChannels * mOutputHeight * mOutputWidth;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalSize + threadsPerBlock - 1) / threadsPerBlock;
    
    if (mType == PoolType_MAXPOOL) {
        MaxPoolKernel<<<blocksPerGrid, threadsPerBlock>>>(
            (const float*)input, (float*)output,
            mBatch, mChannels,
            mInputHeight, mInputWidth,
            mOutputHeight, mOutputWidth,
            mKernels[0], mKernels[1],
            mStrides[0], mStrides[1],
            mPads[0], mPads[1]
        );
    } else {
        AvgPoolKernel<<<blocksPerGrid, threadsPerBlock>>>(
            (const float*)input, (float*)output,
            mBatch, mChannels,
            mInputHeight, mInputWidth,
            mOutputHeight, mOutputWidth,
            mKernels[0], mKernels[1],
            mStrides[0], mStrides[1],
            mPads[0], mPads[1]
        );
    }
    
    // Check for kernel launch errors
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        MNN_ERROR("MUSA Pool kernel launch failed: %s\n", musaGetErrorString(err));
    }
    
    // Synchronize to ensure completion
    mRuntime->device_sync();
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end PoolExecution onExecute...\n");
#endif
    
    return NO_ERROR;
}

// Creator for Pool operations
class PoolCreator : public MusaBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (op->type() == OpType_Pooling) {
            auto pool = op->main_as_Pool();
            std::vector<int> kernels(2, pool->kernelX());
            std::vector<int> strides(2, pool->strideX());
            std::vector<int> pads(2, pool->padX());
            
            PoolType type = pool->type();
            return new PoolExecution(type, kernels, strides, pads, backend);
        }
        return nullptr;
    }
};

MusaCreatorRegister<PoolCreator> __PoolExecution(OpType_Pooling);

} // namespace MUSA
} // namespace MNN
