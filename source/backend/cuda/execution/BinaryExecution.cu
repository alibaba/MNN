#include "BinaryExecution.hpp"
namespace MNN {
namespace CUDA {

template <typename T>
__global__ void ADDE(const T *input0, const T* input1, T *output, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        output[i] = input0[i] + input1[i];
    }
    return;
}
template <typename T>
__global__ void SUBE(const T *input0, const T* input1, T *output, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        output[i] = input0[i] - input1[i];
    }
    return;
}
template <typename T>
__global__ void MULE(const T *input0, const T* input1, T *output, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        output[i] = input0[i] * input1[i];
    }
    return;
}

template <typename T>
__global__ void ADD(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        output[i] = input0[i * s0] + input1[i * s1];
    }
    return;
}
template <typename T>
__global__ void SUB(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        output[i] = input0[i * s0] - input1[i * s1];
    }
    return;
}
template <typename T>
__global__ void MUL(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        output[i] = input0[i * s0] * input1[i * s1];
    }
    return;
}
template <typename T>
__global__ void DIV(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int sgn = input1[i * s1] > 0 ? 1 : (input1[i * s1] < 0 ? -1 : 0);
        output[i] = sgn * input0[i * s0] / max(abs((float)input1[i * s1]), 0.0000001);
    }
    return;
}
template <typename T>
__global__ void REALDIV(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int sgn = input1[i * s1] > 0 ? 1 : (input1[i * s1] < 0 ? -1 : 0);
        output[i] = sgn * input0[i * s0] / max(abs((float)input1[i * s1]), 0.0000001);
    }
    return;
}
template <typename T>
__global__ void MINIMUM(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        output[i] = min(input0[i * s0], input1[i * s1]);
    }
    return;
}
template <typename T>
__global__ void MAXIMUM(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        output[i] = max(input0[i * s0], input1[i * s1]);
    }
    return;
}
template <typename T>
__global__ void GREATER(const T *input0, const T* input1, int *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        output[i] = input0[i * s0] > input1[i * s1] ? 1 : 0;
    }
    return;
}

template <typename T>
__global__ void LESS(const T *input0, const T* input1, int *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        output[i] = input0[i * s0] < input1[i * s1] ? 1 : 0;
    }
    return;
}

template <typename T>
__global__ void LESS_EQUAL(const T *input0, const T* input1, int *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        output[i] = input0[i * s0] <= input1[i * s1] ? 1 : 0;
    }
    return;
}

template <typename T>
__global__ void GREATER_EQUAL(const T *input0, const T* input1, int *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        output[i] = input0[i * s0] >= input1[i * s1] ? 1 : 0;
    }
    return;
}

template <typename T>
__global__ void EQUAL(const T *input0, const T* input1, int *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        output[i] = input0[i * s0] == input1[i * s1] ? 1 : 0;
    }
    return;
}

template <typename T>
__global__ void NOTEQUAL(const T *input0, const T* input1, int *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        output[i] = input0[i * s0] != input1[i * s1] ? 1 : 0;
    }
    return;
}
template <typename T>
__global__ void FLOORDIV(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int sgn = input1[i * s1] > 0 ? 1 : (input1[i * s1] < 0 ? -1 : 0);
        output[i] = floor(1.0*sgn * input0[i * s0] / max(abs((float)input1[i * s1]), 0.0000001));

    }
    return;
}

template <typename T>
__global__ void FLOORMOD(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        T x = input0[i * s0];
        T y = input1[i * s1];
        int sgn = y > 0 ? 1 : (y < 0 ? -1 : 0);
        T tmp = floor(1.0*sgn * x / max((float)abs(y), 0.0000001));

        output[i] = x - tmp * y;
    }
    return;
}

template <typename T>
__global__ void SquaredDifference(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        T x = input0[i * s0];
        T y = input1[i * s1];
        output[i] = (x - y) * (x - y);
    }
    return;
}

template <typename T>
__global__ void POW(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        T x = input0[i * s0];
        T y = input1[i * s1];
        output[i] = pow(x, y);
    }
    return;
}

template <typename T>
__global__ void ATAN2(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        T x = input0[i * s0];
        T y = input1[i * s1];
        output[i] = atan2(x, y);
    }
    return;
}

template <typename T>
__global__ void MOD(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        T x = input0[i * s0];
        T y = input1[i * s1];
        output[i] = x - x / y;
    }
    return;
}

template <typename T>
__global__ void LOGICALOR(const T *input0, const T* input1, T *output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        T x = input0[i * s0];
        T y = input1[i * s1];
        output[i] = (x || y) ? 1 : 0;
    }
    return;
}
BinaryExecution::BinaryExecution(int opType, Backend *backend) : Execution(backend) {
    mType = opType;
}
BinaryExecution::~BinaryExecution(){
    // Do nothing
}
ErrorCode BinaryExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto count = CUDABackend::realSize(outputs[0]);
    auto inputS0 = CUDABackend::realSize(inputs[0]);
    auto inputS1 = CUDABackend::realSize(inputs[1]);
    int s0 = inputS0 == 1 ? 0 : 1;
    int s1 = inputS1 == 1 ? 0 : 1;
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num() > count ? count : runtime->threads_num();
    //printf("%d - %d\n", block_num, threads_num);
    auto computeFunction = [&](Tensor* input0T, Tensor* input1T, Tensor* outputT) {
        auto input0 = (void*)input0T->deviceId();
        auto input1 = (void*)input1T->deviceId();
        auto output = (void*)outputT->deviceId();

        auto dataType = outputs[0]->getType();
    #define COMPUTE_FLOAT(TYPE, DSTTYPE)\
        if (mType == MNN::BinaryOpOperation_##TYPE ) { TYPE<<<block_num, threads_num>>>((float*)input0, (float*)input1, (DSTTYPE*)output, count, s0, s1); return NO_ERROR;};
    #define COMPUTE_INT(TYPE, DSTTYPE)\
        if (mType == MNN::BinaryOpOperation_##TYPE ) { TYPE<<<block_num, threads_num>>>((int*)input0, (int*)input1, (DSTTYPE*)output, count, s0, s1); return NO_ERROR;};

        if (dataType == halide_type_of<float>()) {
            if (mType == MNN::BinaryOpOperation_ADD && 1 == s0 && 1 == s1) {
                { ADDE<<<block_num, threads_num>>>((float*)input0, (float*)input1, (float*)output, count); return NO_ERROR;};
            }
            if (mType == MNN::BinaryOpOperation_SUB && 1 == s0 && 1 == s1) {
                { SUBE<<<block_num, threads_num>>>((float*)input0, (float*)input1, (float*)output, count); return NO_ERROR;};
            }
            if (mType == MNN::BinaryOpOperation_MUL && 1 == s0 && 1 == s1) {
                { MULE<<<block_num, threads_num>>>((float*)input0, (float*)input1, (float*)output, count); return NO_ERROR;};
            }
            COMPUTE_FLOAT(ADD, float);
            COMPUTE_FLOAT(SUB, float);
            COMPUTE_FLOAT(MUL, float);
            COMPUTE_FLOAT(DIV, float);
            COMPUTE_FLOAT(REALDIV, float);
            COMPUTE_FLOAT(MINIMUM, float);
            COMPUTE_FLOAT(MAXIMUM, float);
            COMPUTE_FLOAT(GREATER, int);
            COMPUTE_FLOAT(LESS, int);
            COMPUTE_FLOAT(LESS_EQUAL, int);
            COMPUTE_FLOAT(GREATER_EQUAL, int);
            COMPUTE_FLOAT(EQUAL, int);
            COMPUTE_FLOAT(NOTEQUAL, int);
            COMPUTE_FLOAT(FLOORDIV, float);
            COMPUTE_FLOAT(FLOORMOD, float);
            COMPUTE_FLOAT(POW, float);
            COMPUTE_FLOAT(SquaredDifference, float);
            COMPUTE_FLOAT(ATAN2, float);
            COMPUTE_FLOAT(MOD, float);
        } else {
            COMPUTE_INT(ADD, int);
            COMPUTE_INT(SUB, int);
            COMPUTE_INT(MUL, int);
            COMPUTE_INT(DIV, int);
            COMPUTE_INT(REALDIV, int);
            COMPUTE_INT(MINIMUM, int);
            COMPUTE_INT(MAXIMUM, int);
            COMPUTE_INT(GREATER, int);
            COMPUTE_INT(LESS, int);
            COMPUTE_INT(LESS_EQUAL, int);
            COMPUTE_INT(GREATER_EQUAL, int);
            COMPUTE_INT(EQUAL, int);
            COMPUTE_INT(NOTEQUAL, int);
            COMPUTE_INT(FLOORDIV, int);
            COMPUTE_INT(FLOORMOD, int);
            COMPUTE_INT(SquaredDifference, int);
            COMPUTE_INT(MOD, int);
            COMPUTE_INT(LOGICALOR, int);
        }
    };
    computeFunction(inputs[0], inputs[1], outputs[0]);
    for (int i=2; i<inputs.size(); ++i) {
        computeFunction(outputs[0], inputs[i], outputs[0]);
    }
    return NO_ERROR;
}
class BinaryCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (op->type() == OpType_BinaryOp) {
            return new BinaryExecution(op->main_as_BinaryOp()->opType(), backend);
        }
        if (op->type() == OpType_Eltwise) {
            switch (op->main_as_Eltwise()->type()) {
                case EltwiseType_SUM:
                    return new BinaryExecution(BinaryOpOperation_ADD, backend);
                case EltwiseType_PROD:
                    return new BinaryExecution(BinaryOpOperation_MUL, backend);
                case EltwiseType_MAXIMUM:
                    return new BinaryExecution(BinaryOpOperation_MAXIMUM, backend);
                default:
                    break;
            }
        }
        return nullptr;
    }
};

static CUDACreatorRegister<BinaryCreator> __init(OpType_BinaryOp);
static CUDACreatorRegister<BinaryCreator> __init2(OpType_Eltwise);
}
}