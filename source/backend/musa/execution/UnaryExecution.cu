//
//  UnaryExecution.cu
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "UnaryExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/musa/core/MusaBackend.hpp"
#include <musa_runtime.h>

namespace MNN {
namespace MUSA {

// MUSA kernel for unary operations
__global__ void UnaryKernel(const float* input, float* output, size_t count, int opType) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    
    float x = input[index];
    float y = 0.0f;
    
    switch (opType) {
        case 0: // SIGMOID
            y = 1.0f / (1.0f + expf(-x));
            break;
        case 1: // TANH
            y = tanhf(x);
            break;
        case 2: // RELU
            y = x > 0 ? x : 0;
            break;
        case 3: // RELU6
            y = x > 0 ? (x < 6 ? x : 6) : 0;
            break;
        default:
            y = x;
            break;
    }
    
    output[index] = y;
}

void callUnary(void* input, void* output, size_t count, MNN::MusaRuntime* runtime, int op_type) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    
    if (blocksPerGrid > 65535) {
        blocksPerGrid = 65535;
    }
    
    UnaryKernel<<<blocksPerGrid, threadsPerBlock>>>((const float*)input, (float*)output, count, op_type);
    
    // Check for kernel launch errors
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        MNN_ERROR("MUSA kernel launch failed: %s\n", musaGetErrorString(err));
    }
    
    // Synchronize to ensure completion
    runtime->device_sync();
}

UnaryExecution::UnaryExecution(UnaryOpOperation opType, Backend* backend) : Execution(backend) {
    auto musaBackend = static_cast<MusaBackend*>(backend);
    mRuntime = musaBackend->getMusaRuntime();
    mOpType = opType;
}

ErrorCode UnaryExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mCount = MusaBackend::realSize(inputs[0]);
    return NO_ERROR;
}

ErrorCode UnaryExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start UnaryExecution onExecute...\n");
#endif
    
    auto input = inputs[0]->deviceId();
    auto output = outputs[0]->deviceId();
    
    callUnary((void*)input, (void*)output, mCount, mRuntime, mOpType);
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end UnaryExecution onExecute...\n");
#endif
    
    return NO_ERROR;
}

// Creator for Unary operations
class UnaryCreator : public MusaBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (op->type() == OpType_UnaryOp) {
            return new UnaryExecution(op->main_as_UnaryOp()->opType(), backend);
        }
        if (op->type() == OpType_Sigmoid) {
            return new UnaryExecution(UnaryOpOperation_SIGMOID, backend);
        }
        if (op->type() == OpType_TanH) {
            return new UnaryExecution(UnaryOpOperation_TANH, backend);
        }
        if (op->type() == OpType_ReLU) {
            return new UnaryExecution(UnaryOpOperation_RELU, backend);
        }
        if (op->type() == OpType_ReLU6) {
            return new UnaryExecution(UnaryOpOperation_RELU6, backend);
        }
        return nullptr;
    }
};

MusaCreatorRegister<UnaryCreator> __UnaryExecution(OpType_UnaryOp);
MusaCreatorRegister<UnaryCreator> __SigmoidExecution(OpType_Sigmoid);
MusaCreatorRegister<UnaryCreator> __TanhExecution(OpType_TanH);
MusaCreatorRegister<UnaryCreator> __ReluExecution(OpType_ReLU);
MusaCreatorRegister<UnaryCreator> __Relu6Execution(OpType_ReLU6);

} // namespace MUSA
} // namespace MNN
