//
//  SliceExecution.cu
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

// MUSA kernel for slice operation
__global__ void SliceKernel(const float* input, float* output,
                            const int* starts, const int* sizes,
                            int ndim, int totalSize,
                            const int* inputStrides, const int* outputStrides) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= totalSize) return;
    
    // Decode output index to multi-dimensional index
    int tempIdx = idx;
    int multiIdx[8];  // Support up to 8 dimensions
    for (int i = ndim - 1; i >= 0; --i) {
        multiIdx[i] = tempIdx % outputStrides[i];
        tempIdx /= outputStrides[i];
    }
    
    // Apply starts to get input index
    int inputIdx = 0;
    for (int i = 0; i < ndim; ++i) {
        inputIdx += (multiIdx[i] + starts[i]) * inputStrides[i];
    }
    
    output[idx] = input[inputIdx];
}

class SliceExecution : public Execution {
public:
    SliceExecution(const std::vector<int>& starts, const std::vector<int>& sizes, 
                   const std::vector<int>& axes, Backend* backend) 
        : Execution(backend), mStarts(starts), mSizes(sizes), mAxes(axes) {}
    
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        mNdim = inputs[0]->dimensions();
        
        // Calculate input and output strides
        mInputStrides.resize(mNdim);
        mOutputStrides.resize(mNdim);
        
        auto input = inputs[0];
        auto output = outputs[0];
        
        int inputStride = 1;
        int outputStride = 1;
        
        for (int i = mNdim - 1; i >= 0; --i) {
            mInputStrides[i] = inputStride;
            mOutputStrides[i] = outputStride;
            inputStride *= input->length(i);
            outputStride *= output->length(i);
        }
        
        return NO_ERROR;
    }
    
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
#ifdef LOG_VERBOSE
        MNN_PRINT("start SliceExecution onExecute...\n");
#endif
        
        auto input = inputs[0];
        auto output = outputs[0];
        
        void* inputPtr = (void*)input->deviceId();
        void* outputPtr = (void*)output->deviceId();
        
        int totalSize = output->elementSize();
        
        // Copy parameters to device
        int* dStarts = nullptr;
        int* dInputStrides = nullptr;
        int* dOutputStrides = nullptr;
        
        musaMalloc(&dStarts, sizeof(int) * mNdim);
        musaMalloc(&dInputStrides, sizeof(int) * mNdim);
        musaMalloc(&dOutputStrides, sizeof(int) * mNdim);
        
        musaMemcpy(dStarts, mStarts.data(), sizeof(int) * mNdim, MNNMemcpyHostToDevice);
        musaMemcpy(dInputStrides, mInputStrides.data(), sizeof(int) * mNdim, MNNMemcpyHostToDevice);
        musaMemcpy(dOutputStrides, mOutputStrides.data(), sizeof(int) * mNdim, MNNMemcpyHostToDevice);
        
        dim3 threadsPerBlock(256);
        dim3 blocksPerGrid((totalSize + 255) / 256);
        
        SliceKernel<<<blocksPerGrid, threadsPerBlock>>>(
            (const float*)inputPtr, (float*)outputPtr,
            dStarts, mSizes.data(), mNdim, totalSize,
            dInputStrides, dOutputStrides);
        
        // Check for kernel launch errors
        musaError_t err = musaGetLastError();
        if (err != musaSuccess) {
            MNN_ERROR("MUSA Slice kernel launch failed: %s\n", musaGetErrorString(err));
        }
        
        // Synchronize to ensure completion
        auto musaBackend = static_cast<MusaBackend*>(backend());
        musaBackend->getMusaRuntime()->device_sync();
        
        // Free temporary device memory
        musaFree(dStarts);
        musaFree(dInputStrides);
        musaFree(dOutputStrides);
        
#ifdef LOG_VERBOSE
        MNN_PRINT("end SliceExecution onExecute...\n");
#endif
        
        return NO_ERROR;
    }
    
private:
    std::vector<int> mStarts;
    std::vector<int> mSizes;
    std::vector<int> mAxes;
    int mNdim;
    std::vector<int> mInputStrides;
    std::vector<int> mOutputStrides;
};

// Creator for Slice operations
class SliceCreator : public MusaBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        std::vector<int> starts, sizes, axes;
        
        if (op->type() == OpType_Slice) {
            auto slice = op->main_as_Slice();
            auto startsVec = slice->starts();
            auto sizesVec = slice->sizes();
            auto axesVec = slice->axes();
            
            for (int i = 0; i < startsVec->size(); ++i) {
                starts.push_back(startsVec->data()[i]);
            }
            for (int i = 0; i < sizesVec->size(); ++i) {
                sizes.push_back(sizesVec->data()[i]);
            }
            if (axesVec != nullptr) {
                for (int i = 0; i < axesVec->size(); ++i) {
                    axes.push_back(axesVec->data()[i]);
                }
            } else {
                for (int i = 0; i < starts.size(); ++i) {
                    axes.push_back(i);
                }
            }
        }
        
        return new SliceExecution(starts, sizes, axes, backend);
    }
};

MusaCreatorRegister<SliceCreator> __SliceExecution(OpType_Slice);

} // namespace MUSA
} // namespace MNN
