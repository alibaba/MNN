//
//  ReduceExecution.cu
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

// MUSA kernel for reduce sum
__global__ void ReduceSumKernel(const float* input, float* output, 
                                int outerSize, int reduceSize, int innerSize) {
    int outerIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int innerIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outerIdx >= outerSize || innerIdx >= innerSize) return;
    
    float sum = 0.0f;
    for (int i = 0; i < reduceSize; ++i) {
        int idx = (outerIdx * reduceSize + i) * innerSize + innerIdx;
        sum += input[idx];
    }
    
    output[outerIdx * innerSize + innerIdx] = sum;
}

// MUSA kernel for reduce max
__global__ void ReduceMaxKernel(const float* input, float* output, 
                                int outerSize, int reduceSize, int innerSize) {
    int outerIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int innerIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outerIdx >= outerSize || innerIdx >= innerSize) return;
    
    float maxVal = -FLT_MAX;
    for (int i = 0; i < reduceSize; ++i) {
        int idx = (outerIdx * reduceSize + i) * innerSize + innerIdx;
        maxVal = fmaxf(maxVal, input[idx]);
    }
    
    output[outerIdx * innerSize + innerIdx] = maxVal;
}

// MUSA kernel for reduce min
__global__ void ReduceMinKernel(const float* input, float* output, 
                                int outerSize, int reduceSize, int innerSize) {
    int outerIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int innerIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outerIdx >= outerSize || innerIdx >= innerSize) return;
    
    float minVal = FLT_MAX;
    for (int i = 0; i < reduceSize; ++i) {
        int idx = (outerIdx * reduceSize + i) * innerSize + innerIdx;
        minVal = fminf(minVal, input[idx]);
    }
    
    output[outerIdx * innerSize + innerIdx] = minVal;
}

// MUSA kernel for reduce mean
__global__ void ReduceMeanKernel(const float* input, float* output, 
                                 int outerSize, int reduceSize, int innerSize) {
    int outerIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int innerIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outerIdx >= outerSize || innerIdx >= innerSize) return;
    
    float sum = 0.0f;
    for (int i = 0; i < reduceSize; ++i) {
        int idx = (outerIdx * reduceSize + i) * innerSize + innerIdx;
        sum += input[idx];
    }
    
    output[outerIdx * innerSize + innerIdx] = sum / reduceSize;
}

class ReduceExecution : public Execution {
public:
    ReduceExecution(ReduceType type, const std::vector<int>& dim, bool keepDims, Backend* backend) 
        : Execution(backend), mType(type), mDim(dim), mKeepDims(keepDims) {}
    
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto input = inputs[0];
        auto output = outputs[0];
        
        // Calculate outer, reduce, and inner sizes
        mOuterSize = 1;
        mReduceSize = 1;
        mInnerSize = 1;
        
        int ndim = input->dimensions();
        
        if (mDim.empty()) {
            // Reduce all dimensions
            mOuterSize = 1;
            mReduceSize = input->elementSize();
            mInnerSize = 1;
        } else {
            // Calculate sizes based on reduce dimensions
            std::vector<bool> isReduced(ndim, false);
            for (int d : mDim) {
                int dim = d < 0 ? d + ndim : d;
                if (dim >= 0 && dim < ndim) {
                    isReduced[dim] = true;
                }
            }
            
            // Simple case: reduce contiguous dimensions
            // For more complex cases, we need a more sophisticated approach
            for (int i = 0; i < ndim; ++i) {
                if (isReduced[i]) {
                    mReduceSize *= input->length(i);
                } else {
                    if (mOuterSize == 1 && mReduceSize > 1) {
                        mOuterSize *= input->length(i);
                    } else {
                        mInnerSize *= input->length(i);
                    }
                }
            }
        }
        
        return NO_ERROR;
    }
    
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
#ifdef LOG_VERBOSE
        MNN_PRINT("start ReduceExecution onExecute...\n");
#endif
        
        auto input = inputs[0];
        auto output = outputs[0];
        
        void* inputPtr = (void*)input->deviceId();
        void* outputPtr = (void*)output->deviceId();
        
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((mInnerSize + 15) / 16, (mOuterSize + 15) / 16);
        
        switch (mType) {
            case ReduceType_SUM:
                ReduceSumKernel<<<blocksPerGrid, threadsPerBlock>>>(
                    (const float*)inputPtr, (float*)outputPtr, mOuterSize, mReduceSize, mInnerSize);
                break;
            case ReduceType_MAX:
                ReduceMaxKernel<<<blocksPerGrid, threadsPerBlock>>>(
                    (const float*)inputPtr, (float*)outputPtr, mOuterSize, mReduceSize, mInnerSize);
                break;
            case ReduceType_MIN:
                ReduceMinKernel<<<blocksPerGrid, threadsPerBlock>>>(
                    (const float*)inputPtr, (float*)outputPtr, mOuterSize, mReduceSize, mInnerSize);
                break;
            case ReduceType_MEAN:
                ReduceMeanKernel<<<blocksPerGrid, threadsPerBlock>>>(
                    (const float*)inputPtr, (float*)outputPtr, mOuterSize, mReduceSize, mInnerSize);
                break;
            default:
                ReduceSumKernel<<<blocksPerGrid, threadsPerBlock>>>(
                    (const float*)inputPtr, (float*)outputPtr, mOuterSize, mReduceSize, mInnerSize);
                break;
        }
        
        // Check for kernel launch errors
        musaError_t err = musaGetLastError();
        if (err != musaSuccess) {
            MNN_ERROR("MUSA Reduce kernel launch failed: %s\n", musaGetErrorString(err));
        }
        
        // Synchronize to ensure completion
        auto musaBackend = static_cast<MusaBackend*>(backend());
        musaBackend->getMusaRuntime()->device_sync();
        
#ifdef LOG_VERBOSE
        MNN_PRINT("end ReduceExecution onExecute...\n");
#endif
        
        return NO_ERROR;
    }
    
private:
    ReduceType mType;
    std::vector<int> mDim;
    bool mKeepDims;
    int mOuterSize;
    int mReduceSize;
    int mInnerSize;
};

// Creator for Reduce operations
class ReduceCreator : public MusaBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        ReduceType type = ReduceType_SUM;
        bool keepDims = false;
        std::vector<int> dim;
        
        if (op->type() == OpType_ReduceSum) {
            type = ReduceType_SUM;
            if (op->main_as_Axis() != nullptr) {
                dim.push_back(op->main_as_Axis()->axis());
            }
            keepDims = op->main_as_Axis() != nullptr && op->main_as_Axis()->keepDims();
        } else if (op->type() == OpType_ReduceMax) {
            type = ReduceType_MAX;
        } else if (op->type() == OpType_ReduceMin) {
            type = ReduceType_MIN;
        } else if (op->type() == OpType_ReduceMean) {
            type = ReduceType_MEAN;
        }
        
        return new ReduceExecution(type, dim, keepDims, backend);
    }
};

MusaCreatorRegister<ReduceCreator> __ReduceSumExecution(OpType_ReduceSum);
MusaCreatorRegister<ReduceCreator> __ReduceMaxExecution(OpType_ReduceMax);
MusaCreatorRegister<ReduceCreator> __ReduceMinExecution(OpType_ReduceMin);
MusaCreatorRegister<ReduceCreator> __ReduceMeanExecution(OpType_ReduceMean);

} // namespace MUSA
} // namespace MNN
