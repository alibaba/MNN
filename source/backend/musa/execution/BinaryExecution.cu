//
//  BinaryExecution.cu
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "BinaryExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/musa/core/MusaBackend.hpp"
#include <musa_runtime.h>

namespace MNN {
namespace MUSA {

// MUSA kernel for binary operations
__global__ void BinaryKernel(const float* input0, const float* input1, float* output, size_t count, int opType) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    
    float x = input0[index];
    float y = input1[index];
    float result = 0.0f;
    
    switch (opType) {
        case 0: // ADD
            result = x + y;
            break;
        case 1: // SUB
            result = x - y;
            break;
        case 2: // MUL
            result = x * y;
            break;
        case 3: // DIV
            result = x / y;
            break;
        case 4: // POW
            result = powf(x, y);
            break;
        case 5: // MAX
            result = fmaxf(x, y);
            break;
        case 6: // MIN
            result = fminf(x, y);
            break;
        default:
            result = x;
            break;
    }
    
    output[index] = result;
}

void callBinary(void* input0, void* input1, void* output, size_t count, MNN::MusaRuntime* runtime, int op_type) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    
    if (blocksPerGrid > 65535) {
        blocksPerGrid = 65535;
    }
    
    BinaryKernel<<<blocksPerGrid, threadsPerBlock>>>((const float*)input0, (const float*)input1, (float*)output, count, op_type);
    
    // Check for kernel launch errors
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        MNN_ERROR("MUSA kernel launch failed: %s\n", musaGetErrorString(err));
    }
    
    // Synchronize to ensure completion
    runtime->device_sync();
}

BinaryExecution::BinaryExecution(BinaryOpOperation opType, Backend* backend) : Execution(backend) {
    auto musaBackend = static_cast<MusaBackend*>(backend);
    mRuntime = musaBackend->getMusaRuntime();
    mOpType = opType;
}

ErrorCode BinaryExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mCount = MusaBackend::realSize(inputs[0]);
    return NO_ERROR;
}

ErrorCode BinaryExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start BinaryExecution onExecute...\n");
#endif
    
    auto input0 = inputs[0]->deviceId();
    auto input1 = inputs[1]->deviceId();
    auto output = outputs[0]->deviceId();
    
    callBinary((void*)input0, (void*)input1, (void*)output, mCount, mRuntime, mOpType);
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end BinaryExecution onExecute...\n");
#endif
    
    return NO_ERROR;
}

// Creator for Binary operations
class BinaryCreator : public MusaBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (op->type() == OpType_BinaryOp) {
            return new BinaryExecution(op->main_as_BinaryOp()->opType(), backend);
        }
        return nullptr;
    }
};

MusaCreatorRegister<BinaryCreator> __BinaryExecution(OpType_BinaryOp);

} // namespace MUSA
} // namespace MNN
