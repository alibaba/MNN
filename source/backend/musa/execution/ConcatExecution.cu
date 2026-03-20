//
//  ConcatExecution.cu
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "core/MusaBackend.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include <musa_runtime.h>
#include <vector>

namespace MNN {
namespace MUSA {

// MUSA kernel for concat operation
__global__ void ConcatKernel(const float** inputs, float* output, 
                             const int* inputOffsets, int numInputs,
                             int concatSize, int outerSize, int innerSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = outerSize * concatSize * innerSize;
    
    if (idx >= totalSize) return;
    
    int innerIdx = idx % innerSize;
    int tempIdx = idx / innerSize;
    int concatIdx = tempIdx % concatSize;
    int outerIdx = tempIdx / concatSize;
    
    // Find which input tensor this element belongs to
    int inputIdx = 0;
    int localConcatIdx = concatIdx;
    for (int i = 0; i < numInputs - 1; ++i) {
        int inputSize = inputOffsets[i + 1] - inputOffsets[i];
        if (localConcatIdx < inputSize) {
            break;
        }
        localConcatIdx -= inputSize;
        inputIdx++;
    }
    
    int inputOffset = inputOffsets[inputIdx];
    int srcIdx = (outerIdx * (inputOffsets[inputIdx + 1] - inputOffsets[inputIdx]) + localConcatIdx) * innerSize + innerIdx;
    int dstIdx = idx;
    
    output[dstIdx] = inputs[inputIdx][srcIdx];
}

// Simplified concat kernel for single dimension concat
__global__ void ConcatSimpleKernel(const float** inputs, float* output,
                                   const int* inputSizes, int numInputs,
                                   int totalSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= totalSize) return;
    
    int offset = 0;
    for (int i = 0; i < numInputs; ++i) {
        if (idx < offset + inputSizes[i]) {
            output[idx] = inputs[i][idx - offset];
            return;
        }
        offset += inputSizes[i];
    }
}

class ConcatExecution : public Execution {
public:
    ConcatExecution(int axis, Backend* backend) : Execution(backend), mAxis(axis) {}
    
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        mInputs.resize(inputs.size());
        mInputSizes.resize(inputs.size());
        
        int concatDim = 1;
        for (size_t i = 0; i < inputs.size(); ++i) {
            mInputs[i] = inputs[i];
            mInputSizes[i] = inputs[i]->length(mAxis);
            concatDim += inputs[i]->length(mAxis);
        }
        
        return NO_ERROR;
    }
    
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
#ifdef LOG_VERBOSE
        MNN_PRINT("start ConcatExecution onExecute...\n");
#endif
        
        auto output = outputs[0];
        
        // Collect input device pointers
        std::vector<void*> inputPtrs(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            inputPtrs[i] = (void*)inputs[i]->deviceId();
        }
        void* outputPtr = (void*)output->deviceId();
        
        // Copy device pointers to device memory
        const float** dInputs = nullptr;
        int* dInputSizes = nullptr;
        size_t ptrSize = sizeof(float*) * inputs.size();
        size_t sizeSize = sizeof(int) * inputs.size();
        
        musaMalloc(&dInputs, ptrSize);
        musaMalloc(&dInputSizes, sizeSize);
        
        musaMemcpy(dInputs, inputPtrs.data(), ptrSize, MNNMemcpyHostToDevice);
        musaMemcpy(dInputSizes, mInputSizes.data(), sizeSize, MNNMemcpyHostToDevice);
        
        int totalSize = output->elementSize();
        dim3 threadsPerBlock(256);
        dim3 blocksPerGrid((totalSize + 255) / 256);
        
        ConcatSimpleKernel<<<blocksPerGrid, threadsPerBlock>>>(
            dInputs, (float*)outputPtr, dInputSizes, inputs.size(), totalSize);
        
        // Check for kernel launch errors
        musaError_t err = musaGetLastError();
        if (err != musaSuccess) {
            MNN_ERROR("MUSA Concat kernel launch failed: %s\n", musaGetErrorString(err));
        }
        
        // Synchronize to ensure completion
        auto musaBackend = static_cast<MusaBackend*>(backend());
        musaBackend->getMusaRuntime()->device_sync();
        
        // Free temporary device memory
        musaFree(dInputs);
        musaFree(dInputSizes);
        
#ifdef LOG_VERBOSE
        MNN_PRINT("end ConcatExecution onExecute...\n");
#endif
        
        return NO_ERROR;
    }
    
private:
    int mAxis;
    std::vector<Tensor*> mInputs;
    std::vector<int> mInputSizes;
};

// Creator for Concat operations
class ConcatCreator : public MusaBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        int axis = 1;
        if (op->type() == OpType_Concat) {
            axis = op->main_as_Axis()->axis();
        }
        return new ConcatExecution(axis, backend);
    }
};

MusaCreatorRegister<ConcatCreator> __ConcatExecution(OpType_Concat);

} // namespace MUSA
} // namespace MNN
