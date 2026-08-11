//
//  SoftmaxExecution.cu
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "SoftmaxExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/musa/core/MusaBackend.hpp"
#include <musa_runtime.h>

namespace MNN {
namespace MUSA {

// MUSA kernel for softmax operation
__global__ void SoftmaxKernel(const float* input, float* output, int outerCount, int depth, int innerCount) {
    int outerIdx = blockIdx.x;
    int innerIdx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (outerIdx >= outerCount || innerIdx >= innerCount) return;
    
    const float* inPtr = input + outerIdx * depth * innerCount;
    float* outPtr = output + outerIdx * depth * innerCount;
    
    // Find max value for numerical stability
    float maxVal = -FLT_MAX;
    for (int i = 0; i < depth; i++) {
        float val = inPtr[i * innerCount + innerIdx];
        if (val > maxVal) {
            maxVal = val;
        }
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < depth; i++) {
        float expVal = expf(inPtr[i * innerCount + innerIdx] - maxVal);
        outPtr[i * innerCount + innerIdx] = expVal;
        sum += expVal;
    }
    
    // Normalize
    float invSum = 1.0f / sum;
    for (int i = 0; i < depth; i++) {
        outPtr[i * innerCount + innerIdx] *= invSum;
    }
}

SoftmaxExecution::SoftmaxExecution(int axis, Backend* backend) : Execution(backend) {
    auto musaBackend = static_cast<MusaBackend*>(backend);
    mRuntime = musaBackend->getMusaRuntime();
    mAxis = axis;
}

ErrorCode SoftmaxExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto shape = input->shape();
    int dims = shape.size();
    
    if (mAxis < 0) {
        mAxis = dims + mAxis;
    }
    
    mOuterCount = 1;
    for (int i = 0; i < mAxis; i++) {
        mOuterCount *= shape[i];
    }
    
    mDepth = shape[mAxis];
    
    mInnerCount = 1;
    for (int i = mAxis + 1; i < dims; i++) {
        mInnerCount *= shape[i];
    }
    
    return NO_ERROR;
}

ErrorCode SoftmaxExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start SoftmaxExecution onExecute...\n");
#endif
    
    auto input = inputs[0]->deviceId();
    auto output = outputs[0]->deviceId();
    
    int threadsPerBlock = 256;
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(mOuterCount, (mInnerCount + threadsPerBlock - 1) / threadsPerBlock);
    
    SoftmaxKernel<<<gridDim, blockDim>>>((const float*)input, (float*)output, mOuterCount, mDepth, mInnerCount);
    
    // Check for kernel launch errors
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        MNN_ERROR("MUSA Softmax kernel launch failed: %s\n", musaGetErrorString(err));
    }
    
    // Synchronize to ensure completion
    mRuntime->device_sync();
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end SoftmaxExecution onExecute...\n");
#endif
    
    return NO_ERROR;
}

// Creator for Softmax operations
class SoftmaxCreator : public MusaBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        int axis = 1;
        if (op->type() == OpType_Softmax) {
            auto softmax = op->main_as_Softmax();
            if (softmax != nullptr && softmax->axis() != -1) {
                axis = softmax->axis();
            }
            return new SoftmaxExecution(axis, backend);
        }
        return nullptr;
    }
};

MusaCreatorRegister<SoftmaxCreator> __SoftmaxExecution(OpType_Softmax);

} // namespace MUSA
} // namespace MNN
