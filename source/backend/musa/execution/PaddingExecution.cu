//
//  PaddingExecution.cu
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "core/MusaBackend.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include <musa_runtime.h>

namespace MNN {
namespace MUSA {

// MUSA kernel for padding operation
__global__ void PaddingKernel(const float* input, float* output,
                              int batch, int channels, int inHeight, int inWidth,
                              int outHeight, int outWidth,
                              int padTop, int padLeft,
                              float padValue) {
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outX >= outWidth || outY >= outHeight) return;
    
    int inY = outY - padTop;
    int inX = outX - padLeft;
    
    float value = padValue;
    if (inY >= 0 && inY < inHeight && inX >= 0 && inX < inWidth) {
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channels; ++c) {
                int inIdx = ((b * channels + c) * inHeight + inY) * inWidth + inX;
                int outIdx = ((b * channels + c) * outHeight + outY) * outWidth + outX;
                output[outIdx] = input[inIdx];
            }
        }
    } else {
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channels; ++c) {
                int outIdx = ((b * channels + c) * outHeight + outY) * outWidth + outX;
                output[outIdx] = padValue;
            }
        }
    }
}

// Simplified padding kernel for single channel
__global__ void PaddingSimpleKernel(const float* input, float* output,
                                    int totalSize, int inHeight, int inWidth,
                                    int outHeight, int outWidth,
                                    int padTop, int padLeft,
                                    float padValue) {
    int outIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outIdx >= totalSize) return;
    
    int outY = (outIdx / outWidth) % outHeight;
    int outX = outIdx % outWidth;
    
    int inY = outY - padTop;
    int inX = outX - padLeft;
    
    if (inY >= 0 && inY < inHeight && inX >= 0 && inX < inWidth) {
        int inIdx = (outIdx / (outHeight * outWidth)) * (inHeight * inWidth) + 
                    inY * inWidth + inX;
        output[outIdx] = input[inIdx];
    } else {
        output[outIdx] = padValue;
    }
}

class PaddingExecution : public Execution {
public:
    PaddingExecution(const std::vector<int>& pads, float padValue, Backend* backend) 
        : Execution(backend), mPads(pads), mPadValue(padValue) {}
    
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto input = inputs[0];
        auto output = outputs[0];
        
        auto inputShape = input->shape();
        auto outputShape = output->shape();
        
        mBatch = inputShape[0];
        mChannels = inputShape[1];
        mInHeight = inputShape[2];
        mInWidth = inputShape[3];
        
        mOutHeight = outputShape[2];
        mOutWidth = outputShape[3];
        
        mPadTop = mPads[0];
        mPadLeft = mPads[1];
        
        return NO_ERROR;
    }
    
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
#ifdef LOG_VERBOSE
        MNN_PRINT("start PaddingExecution onExecute...\n");
#endif
        
        auto input = inputs[0];
        auto output = outputs[0];
        
        void* inputPtr = (void*)input->deviceId();
        void* outputPtr = (void*)output->deviceId();
        
        int totalSize = output->elementSize();
        
        dim3 threadsPerBlock(256);
        dim3 blocksPerGrid((totalSize + 255) / 256);
        
        PaddingSimpleKernel<<<blocksPerGrid, threadsPerBlock>>>(
            (const float*)inputPtr, (float*)outputPtr,
            totalSize, mInHeight, mInWidth,
            mOutHeight, mOutWidth,
            mPadTop, mPadLeft,
            mPadValue);
        
        // Check for kernel launch errors
        musaError_t err = musaGetLastError();
        if (err != musaSuccess) {
            MNN_ERROR("MUSA Padding kernel launch failed: %s\n", musaGetErrorString(err));
        }
        
        // Synchronize to ensure completion
        auto musaBackend = static_cast<MusaBackend*>(backend());
        musaBackend->getMusaRuntime()->device_sync();
        
#ifdef LOG_VERBOSE
        MNN_PRINT("end PaddingExecution onExecute...\n");
#endif
        
        return NO_ERROR;
    }
    
private:
    std::vector<int> mPads;
    float mPadValue;
    int mBatch;
    int mChannels;
    int mInHeight;
    int mInWidth;
    int mOutHeight;
    int mOutWidth;
    int mPadTop;
    int mPadLeft;
};

// Creator for Padding operations
class PaddingCreator : public MusaBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        std::vector<int> pads;
        float padValue = 0.0f;
        
        if (op->type() == OpType_Padding) {
            auto paddings = op->main_as_Padding();
            auto padList = paddings->pads();
            for (int i = 0; i < padList->size(); ++i) {
                pads.push_back(padList->data()[i]);
            }
            padValue = paddings->value();
        }
        
        return new PaddingExecution(pads, padValue, backend);
    }
};

MusaCreatorRegister<PaddingCreator> __PaddingExecution(OpType_Padding);

} // namespace MUSA
} // namespace MNN
