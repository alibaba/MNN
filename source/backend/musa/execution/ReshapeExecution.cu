//
//  ReshapeExecution.cu
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

// MUSA kernel for reshape (copy with shape change)
__global__ void ReshapeKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    
    output[idx] = input[idx];
}

class ReshapeExecution : public Execution {
public:
    ReshapeExecution(Backend* backend) : Execution(backend) {}
    
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        // Reshape doesn't change data, just the shape
        return NO_ERROR;
    }
    
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
#ifdef LOG_VERBOSE
        MNN_PRINT("start ReshapeExecution onExecute...\n");
#endif
        
        auto input = inputs[0];
        auto output = outputs[0];
        
        void* inputPtr = (void*)input->deviceId();
        void* outputPtr = (void*)output->deviceId();
        
        int size = input->elementSize();
        
        // If input and output are contiguous, just copy
        if (size > 0) {
            dim3 threadsPerBlock(256);
            dim3 blocksPerGrid((size + 255) / 256);
            
            ReshapeKernel<<<blocksPerGrid, threadsPerBlock>>>(
                (const float*)inputPtr, (float*)outputPtr, size);
            
            // Check for kernel launch errors
            musaError_t err = musaGetLastError();
            if (err != musaSuccess) {
                MNN_ERROR("MUSA Reshape kernel launch failed: %s\n", musaGetErrorString(err));
            }
            
            // Synchronize to ensure completion
            auto musaBackend = static_cast<MusaBackend*>(backend());
            musaBackend->getMusaRuntime()->device_sync();
        }
        
#ifdef LOG_VERBOSE
        MNN_PRINT("end ReshapeExecution onExecute...\n");
#endif
        
        return NO_ERROR;
    }
};

// MUSA kernel for transpose
__global__ void TransposeKernel(const float* input, float* output, 
                                const int* perm, const int* inputStrides, const int* outputStrides,
                                int ndim, int totalSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= totalSize) return;
    
    // Decode output index to multi-dimensional index
    int tempIdx = idx;
    int multiIdx[8];  // Support up to 8 dimensions
    for (int i = ndim - 1; i >= 0; --i) {
        multiIdx[i] = tempIdx % outputStrides[i];
        tempIdx /= outputStrides[i];
    }
    
    // Apply permutation to get input index
    int inputIdx = 0;
    for (int i = 0; i < ndim; ++i) {
        inputIdx += multiIdx[perm[i]] * inputStrides[i];
    }
    
    output[idx] = input[inputIdx];
}

class TransposeExecution : public Execution {
public:
    TransposeExecution(const std::vector<int>& perm, Backend* backend) 
        : Execution(backend), mPerm(perm) {}
    
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        mNdim = inputs[0]->dimensions();
        
        // Calculate input and output strides
        mInputStrides.resize(mNdim);
        mOutputStrides.resize(mNdim);
        
        int inputStride = 1;
        int outputStride = 1;
        
        for (int i = mNdim - 1; i >= 0; --i) {
            mInputStrides[i] = inputStride;
            mOutputStrides[i] = outputStride;
            inputStride *= inputs[0]->length(i);
            outputStride *= outputs[0]->length(i);
        }
        
        return NO_ERROR;
    }
    
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
#ifdef LOG_VERBOSE
        MNN_PRINT("start TransposeExecution onExecute...\n");
#endif
        
        auto input = inputs[0];
        auto output = outputs[0];
        
        void* inputPtr = (void*)input->deviceId();
        void* outputPtr = (void*)output->deviceId();
        
        int totalSize = output->elementSize();
        
        // Copy perm and strides to device
        int* dPerm = nullptr;
        int* dInputStrides = nullptr;
        int* dOutputStrides = nullptr;
        
        musaMalloc(&dPerm, sizeof(int) * mNdim);
        musaMalloc(&dInputStrides, sizeof(int) * mNdim);
        musaMalloc(&dOutputStrides, sizeof(int) * mNdim);
        
        musaMemcpy(dPerm, mPerm.data(), sizeof(int) * mNdim, MNNMemcpyHostToDevice);
        musaMemcpy(dInputStrides, mInputStrides.data(), sizeof(int) * mNdim, MNNMemcpyHostToDevice);
        musaMemcpy(dOutputStrides, mOutputStrides.data(), sizeof(int) * mNdim, MNNMemcpyHostToDevice);
        
        dim3 threadsPerBlock(256);
        dim3 blocksPerGrid((totalSize + 255) / 256);
        
        TransposeKernel<<<blocksPerGrid, threadsPerBlock>>>(
            (const float*)inputPtr, (float*)outputPtr,
            dPerm, dInputStrides, dOutputStrides, mNdim, totalSize);
        
        // Check for kernel launch errors
        musaError_t err = musaGetLastError();
        if (err != musaSuccess) {
            MNN_ERROR("MUSA Transpose kernel launch failed: %s\n", musaGetErrorString(err));
        }
        
        // Synchronize to ensure completion
        auto musaBackend = static_cast<MusaBackend*>(backend());
        musaBackend->getMusaRuntime()->device_sync();
        
        // Free temporary device memory
        musaFree(dPerm);
        musaFree(dInputStrides);
        musaFree(dOutputStrides);
        
#ifdef LOG_VERBOSE
        MNN_PRINT("end TransposeExecution onExecute...\n");
#endif
        
        return NO_ERROR;
    }
    
private:
    std::vector<int> mPerm;
    int mNdim;
    std::vector<int> mInputStrides;
    std::vector<int> mOutputStrides;
};

// Creator for Reshape operations
class ReshapeCreator : public MusaBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new ReshapeExecution(backend);
    }
};

// Creator for Transpose operations
class TransposeCreator : public MusaBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        std::vector<int> perm;
        if (op->type() == OpType_Transpose) {
            auto permVec = op->main_as_Transpose()->perm();
            for (int i = 0; i < permVec->size(); ++i) {
                perm.push_back(permVec->data()[i]);
            }
        }
        return new TransposeExecution(perm, backend);
    }
};

MusaCreatorRegister<ReshapeCreator> __ReshapeExecution(OpType_Reshape);
MusaCreatorRegister<ReshapeCreator> __ReshapeTranspose(OpType_Transpose);

} // namespace MUSA
} // namespace MNN
