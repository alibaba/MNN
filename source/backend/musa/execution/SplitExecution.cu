//
//  SplitExecution.cu
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

// MUSA kernel for split operation
__global__ void SplitKernel(const float* input, float** outputs,
                            const int* outputOffsets, int numOutputs,
                            int splitSize, int outerSize, int innerSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = outerSize * splitSize * innerSize;
    
    if (idx >= totalSize) return;
    
    int innerIdx = idx % innerSize;
    int tempIdx = idx / innerSize;
    int splitIdx = tempIdx % splitSize;
    int outerIdx = tempIdx / splitSize;
    
    // Find which output tensor this element belongs to
    int outputIdx = 0;
    int localSplitIdx = splitIdx;
    for (int i = 0; i < numOutputs - 1; ++i) {
        int outputSize = outputOffsets[i + 1] - outputOffsets[i];
        if (localSplitIdx < outputSize) {
            break;
        }
        localSplitIdx -= outputSize;
        outputIdx++;
    }
    
    int srcIdx = idx;
    int dstIdx = (outerIdx * (outputOffsets[outputIdx + 1] - outputOffsets[outputIdx]) + localSplitIdx) * innerSize + innerIdx;
    
    outputs[outputIdx][dstIdx] = input[srcIdx];
}

class SplitExecution : public Execution {
public:
    SplitExecution(int axis, Backend* backend) : Execution(backend), mAxis(axis) {}
    
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        mOutputs.resize(outputs.size());
        mOutputSizes.resize(outputs.size());
        
        for (size_t i = 0; i < outputs.size(); ++i) {
            mOutputs[i] = outputs[i];
            mOutputSizes[i] = outputs[i]->length(mAxis);
        }
        
        return NO_ERROR;
    }
    
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
#ifdef LOG_VERBOSE
        MNN_PRINT("start SplitExecution onExecute...\n");
#endif
        
        auto input = inputs[0];
        
        // Collect output device pointers
        std::vector<void*> outputPtrs(outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputPtrs[i] = (void*)outputs[i]->deviceId();
        }
        void* inputPtr = (void*)input->deviceId();
        
        // Copy device pointers to device memory
        float** dOutputs = nullptr;
        int* dOutputOffsets = nullptr;
        size_t ptrSize = sizeof(float*) * outputs.size();
        size_t sizeSize = sizeof(int) * (outputs.size() + 1);
        
        musaMalloc(&dOutputs, ptrSize);
        musaMalloc(&dOutputOffsets, sizeSize);
        
        musaMemcpy(dOutputs, outputPtrs.data(), ptrSize, MNNMemcpyHostToDevice);
        
        // Calculate output offsets
        std::vector<int> outputOffsets(outputs.size() + 1, 0);
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputOffsets[i + 1] = outputOffsets[i] + mOutputSizes[i];
        }
        musaMemcpy(dOutputOffsets, outputOffsets.data(), sizeSize, MNNMemcpyHostToDevice);
        
        int totalSize = input->elementSize();
        dim3 threadsPerBlock(256);
        dim3 blocksPerGrid((totalSize + 255) / 256);
        
        SplitKernel<<<blocksPerGrid, threadsPerBlock>>>(
            (const float*)inputPtr, dOutputs, dOutputOffsets, outputs.size(),
            input->length(mAxis), 1, totalSize / input->length(mAxis));
        
        // Check for kernel launch errors
        musaError_t err = musaGetLastError();
        if (err != musaSuccess) {
            MNN_ERROR("MUSA Split kernel launch failed: %s\n", musaGetErrorString(err));
        }
        
        // Synchronize to ensure completion
        auto musaBackend = static_cast<MusaBackend*>(backend());
        musaBackend->getMusaRuntime()->device_sync();
        
        // Free temporary device memory
        musaFree(dOutputs);
        musaFree(dOutputOffsets);
        
#ifdef LOG_VERBOSE
        MNN_PRINT("end SplitExecution onExecute...\n");
#endif
        
        return NO_ERROR;
    }
    
private:
    int mAxis;
    std::vector<Tensor*> mOutputs;
    std::vector<int> mOutputSizes;
};

// Creator for Split operations
class SplitCreator : public MusaBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        int axis = 0;
        if (op->type() == OpType_Split) {
            axis = op->main_as_Split()->axis();
        }
        return new SplitExecution(axis, backend);
    }
};

MusaCreatorRegister<SplitCreator> __SplitExecution(OpType_Split);

} // namespace MUSA
} // namespace MNN
