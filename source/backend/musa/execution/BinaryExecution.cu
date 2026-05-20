//
//  BinaryExecution.cu
//  MNN
//
//  Updated: 2026/02/27 - Fixed binary operations
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "BinaryExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/musa/core/MusaBackend.hpp"
#include <musa_runtime.h>
#include <cmath>

namespace MNN {
namespace MUSA {

// MUSA kernel for binary operations - FIXED
__global__ void BinaryKernel(const float* input0, const float* input1, float* output, size_t count, int opType) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x) {
        float x = input0[index];
        float y = input1[index];
        float result = x; // default: identity
        
        switch (opType) {
            case BinaryOpOperation_ADD:            // 0
                result = x + y;
                break;
            case BinaryOpOperation_SUB:           // 1
                result = x - y;
                break;
            case BinaryOpOperation_MUL:           // 2
                result = x * y;
                break;
            case BinaryOpOperation_DIV:           // 3
                result = x / y;
                break;
            case BinaryOpOperation_MAX_TEMP:      // 4
                result = fmaxf(x, y);
                break;
            case BinaryOpOperation_MIN_TEMP:      // 5
                result = fminf(x, y);
                break;
            case BinaryOpOperation_POW:           // 6
                result = powf(x, y);
                break;
            case BinaryOpOperation_REALDIV:       // 7
                result = x / y;
                break;
            case BinaryOpOperation_MINIMUM:       // 8
                result = fminf(x, y);
                break;
            case BinaryOpOperation_MAXIMUM:        // 9
                result = fmaxf(x, y);
                break;
            case BinaryOpOperation_GREATER:        // 10
                result = (x > y) ? 1.0f : 0.0f;
                break;
            case BinaryOpOperation_GREATER_EQUAL:  // 11
                result = (x >= y) ? 1.0f : 0.0f;
                break;
            case BinaryOpOperation_LESS:           // 12
                result = (x < y) ? 1.0f : 0.0f;
                break;
            case BinaryOpOperation_FLOORDIV:       // 13
                result = floorf(x / y);
                break;
            case BinaryOpOperation_SquaredDifference: // 14
                result = (x - y) * (x - y);
                break;
            case BinaryOpOperation_EQUAL:         // 15
                result = (x == y) ? 1.0f : 0.0f;
                break;
            case BinaryOpOperation_LESS_EQUAL:     // 16
                result = (x <= y) ? 1.0f : 0.0f;
                break;
            case BinaryOpOperation_FLOORMOD:       // 17
                result = fmodf(x, y);
                if (result != 0 && (result < 0) != (y < 0)) {
                    result += y;
                }
                break;
            case BinaryOpOperation_MOD:           // 19
                result = fmodf(x, y);
                break;
            case BinaryOpOperation_ATAN2:         // 20
                result = atan2f(x, y);
                break;
            case BinaryOpOperation_LOGICALOR:      // 21
                result = (x != 0.0f || y != 0.0f) ? 1.0f : 0.0f;
                break;
            case BinaryOpOperation_NOTEQUAL:       // 22
                result = (x != y) ? 1.0f : 0.0f;
                break;
            case BinaryOpOperation_BITWISE_AND:    // 23
                result = (float)((int)x & (int)y);
                break;
            case BinaryOpOperation_BITWISE_OR:     // 24
                result = (float)((int)x | (int)y);
                break;
            case BinaryOpOperation_BITWISE_XOR:    // 25
                result = (float)((int)x ^ (int)y);
                break;
            case BinaryOpOperation_LOGICALXOR:     // 26
                result = ((x != 0.0f) != (y != 0.0f)) ? 1.0f : 0.0f;
                break;
            case BinaryOpOperation_LEFTSHIFT:       // 27
                result = (float)((int)x << (int)y);
                break;
            case BinaryOpOperation_RIGHTSHIFT:      // 28
                result = (float)((int)x >> (int)y);
                break;
            default:
                result = x;
                break;
        }
        
        output[index] = result;
    }
}

void callBinary(void* input0, void* input1, void* output, size_t count, MNN::MusaRuntime* runtime, int op_type) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    
    if (blocksPerGrid > 65535) {
        blocksPerGrid = 65535;
    }
    
    BinaryKernel<<<blocksPerGrid, threadsPerBlock>>>((const float*)input0, (const float*)input1, (float*)output, count, op_type);
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        MNN_ERROR("MUSA kernel launch failed: %s\n", musaGetErrorString(err));
    }
    
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
    auto input0 = inputs[0]->deviceId();
    auto input1 = inputs[1]->deviceId();
    auto output = outputs[0]->deviceId();
    callBinary((void*)input0, (void*)input1, (void*)output, mCount, mRuntime, mOpType);
    return NO_ERROR;
}

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