//
//  UnaryExecution.cu
//  MNN
//
//  Updated: 2026/02/27 - Fixed unary operations
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "UnaryExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/musa/core/MusaBackend.hpp"
#include <musa_runtime.h>
#include <cmath>

namespace MNN {
namespace MUSA {

// MUSA kernel for unary operations - FIXED
__global__ void UnaryKernel(const float* input, float* output, size_t count, int opType) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x) {
        float x = input[index];
        float y = x; // default: identity
        
        switch (opType) {
            case UnaryOpOperation_ABS:        // 0
                y = fabsf(x);
                break;
            case UnaryOpOperation_NEG:        // 1
                y = -x;
                break;
            case UnaryOpOperation_FLOOR:      // 2
                y = floorf(x);
                break;
            case UnaryOpOperation_CEIL:       // 3
                y = ceilf(x);
                break;
            case UnaryOpOperation_SQUARE:     // 4
                y = x * x;
                break;
            case UnaryOpOperation_SQRT:       // 5
                y = sqrtf(x);
                break;
            case UnaryOpOperation_RSQRT:      // 6
                y = rsqrtf(x);
                break;
            case UnaryOpOperation_EXP:        // 7
                y = expf(x);
                break;
            case UnaryOpOperation_LOG:        // 8
                y = logf(x);
                break;
            case UnaryOpOperation_SIN:        // 9
                y = sinf(x);
                break;
            case UnaryOpOperation_COS:        // 10
                y = cosf(x);
                break;
            case UnaryOpOperation_TAN:        // 11
                y = tanf(x);
                break;
            case UnaryOpOperation_ASIN:       // 12
                y = asinf(x);
                break;
            case UnaryOpOperation_ACOS:       // 13
                y = acosf(x);
                break;
            case UnaryOpOperation_ATAN:       // 14
                y = atanf(x);
                break;
            case UnaryOpOperation_RECIPROCAL: // 15
                y = 1.0f / x;
                break;
            case UnaryOpOperation_LOG1P:      // 16
                y = log1pf(x);
                break;
            case UnaryOpOperation_BNLL:       // 17
                y = (x > 0) ? (x + logf(1.0f + expf(-x))) : logf(1.0f + expf(x));
                break;
            case UnaryOpOperation_ACOSH:      // 18
                y = acoshf(x);
                break;
            case UnaryOpOperation_SINH:       // 19
                y = sinhf(x);
                break;
            case UnaryOpOperation_ASINH:      // 20
                y = asinhf(x);
                break;
            case UnaryOpOperation_ATANH:      // 21
                y = atanhf(x);
                break;
            case UnaryOpOperation_SIGN:       // 22
                y = (x > 0) ? 1.0f : ((x < 0) ? -1.0f : 0.0f);
                break;
            case UnaryOpOperation_ROUND:      // 23
                y = roundf(x);
                break;
            case UnaryOpOperation_COSH:       // 24
                y = coshf(x);
                break;
            case UnaryOpOperation_ERF:        // 25
                y = erff(x);
                break;
            case UnaryOpOperation_ERFC:       // 26
                y = erfcf(x);
                break;
            case UnaryOpOperation_ERFINV:     // 27
                y = erfinvf(x);
                break;
            case UnaryOpOperation_EXPM1:      // 28
                y = expm1f(x);
                break;
            case UnaryOpOperation_SIGMOID:    // 29
                y = (x > 87.0f) ? 1.0f : ((x < -87.0f) ? 0.0f : 1.0f / (1.0f + expf(-x)));
                break;
            case UnaryOpOperation_TANH:       // 30
                y = tanhf(x);
                break;
            case UnaryOpOperation_HARDSWISH:  // 31
                y = (1.0f / 6.0f) * x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
                break;
            case UnaryOpOperation_GELU:       // 32
                y = 0.5f * x * (1.0f + tanhf(0.79788458f * (x + 0.044715f * x * x * x)));
                break;
            case UnaryOpOperation_GELU_STANDARD: // 33
                y = 0.5f * x * (1.0f + erff(x * 0.7071067932881648f));
                break;
            case UnaryOpOperation_SILU:       // 34
                y = (x > 87.0f) ? x : ((x < -87.0f) ? 0.0f : x / (1.0f + expf(-x)));
                break;
            default:
                y = x; // identity
                break;
        }
        
        output[index] = y;
    }
}

void callUnary(void* input, void* output, size_t count, MNN::MusaRuntime* runtime, int op_type) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    
    if (blocksPerGrid > 65535) {
        blocksPerGrid = 65535;
    }
    
    UnaryKernel<<<blocksPerGrid, threadsPerBlock>>>((const float*)input, (float*)output, count, op_type);
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        MNN_ERROR("MUSA kernel launch failed: %s\n", musaGetErrorString(err));
    }
    
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
    auto input = inputs[0]->deviceId();
    auto output = outputs[0]->deviceId();
    callUnary((void*)input, (void*)output, mCount, mRuntime, mOpType);
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