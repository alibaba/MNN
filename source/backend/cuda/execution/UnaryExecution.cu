//
//  UnaryExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "UnaryExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/cuda/core/CUDABackend.hpp"
#include <cuda_runtime.h>

namespace MNN {
namespace CUDA {

template <typename T>
__global__ void ABS(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = abs(input[i]);
  }
  return;
}
template <typename T>
__global__ void EXP(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = exp(input[i]);
  }
  return;
}

template <typename T>
__global__ void NEG(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = -input[i];
  }
  return;
}
template <typename T>
__global__ void RECIPROCAL(T *input, T *output, size_t count) {
  T one = 1.0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = one / input[i];
  }
  return;
}

template <typename T>
__global__ void FLOOR(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = floor(input[i]);
  }
}

template <typename T>
__global__ void CEIL(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = ceil(input[i]);
  }
}
template <typename T>
__global__ void SQUARE(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i] * input[i];
  }
  return;
}

template <typename T>
__global__ void SQRT(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = sqrt(input[i]);
  }
  return;
}

template <typename T>
__global__ void RSQRT(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = rsqrt(input[i]);
  }
  return;
}

template <typename T>
__global__ void LOG(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = log(input[i]);
  }
  return;
}
template <typename T>
__global__ void SIN(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = sin(input[i]);
  }
  return;
}

template <typename T>
__global__ void COS(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = cos(input[i]);
  }
  return;
}

template <typename T>
__global__ void TAN(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = tan(input[i]);
  }
  return;
}
template <typename T>
__global__ void ASIN(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = asin(input[i]);
  }
  return;
}
template <typename T>
__global__ void ACOS(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = acos(input[i]);
  }
  return;
}
template <typename T>
__global__ void ATAN(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = atan(input[i]);
  }
  return;
}
template <typename T>
__global__ void LOG1P(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = log(1+input[i]);
  }
  return;
}
template <typename T>
__global__ void TANH(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = tanh(input[i]);
  }
  return;
}
template <typename T>
__global__ void SIGMOID(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = 1. / (1. + exp(-input[i]));
  }
  return;
}

template <typename T>
__global__ void EXPM1(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = exp(input[i]) - 1;
  }
  return;
}
template <typename T>
__global__ void ATANH(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = atanh(input[i]);
  }
  return;
}
template <typename T>
__global__ void ACOSH(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = acosh(input[i]);
  }
  return;
}
template <typename T>
__global__ void SIGN(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    T x = input[i];
    output[i] = x > 0 ? 1 : (x<0 ? -1 : 0);
  }
  return;
}
template <typename T>
__global__ void COSH(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = cosh(input[i]);
  }
  return;
}
template <typename T>
__global__ void ROUND(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = round(input[i]);
  }
  return;
}
template <typename T>
__global__ void SINH(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = sinh(input[i]);
  }
  return;
}
template <typename T>
__global__ void ASINH(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = asinh(input[i]);
  }
  return;
}
template <typename T>
__global__ void HARDSWISH(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    if (input[i] <= -3) {
        output[i] = 0;
    } else if (input[i] >= 3) {
        output[i] = input[i];
    } else {
        output[i] = input[i] * (input[i] + 3) / 6;
    }
  }
  return;
}

void callUnary(void *input, void *output, size_t count, MNN::CUDARuntime* runtime, halide_type_t data_type,
   MNN::UnaryOpOperation op_type)
{
  int block_num = runtime->blocks_num(count);
  int threads_num = runtime->threads_num() > count ? count : runtime->threads_num();
#define COMPUTE(TYPE)\
  if (op_type == MNN::UnaryOpOperation_##TYPE ) { TYPE<<<block_num, threads_num>>>((float*)input, (float*)output, count); return;};

  COMPUTE(ABS);
  COMPUTE(NEG);
  COMPUTE(FLOOR);
  COMPUTE(CEIL);
  COMPUTE(SQUARE);
  COMPUTE(SQRT);
  COMPUTE(RSQRT);
  COMPUTE(EXP);
  COMPUTE(LOG);
  COMPUTE(SIN);
  COMPUTE(COS);
  COMPUTE(TAN);
  COMPUTE(ASIN);
  COMPUTE(ACOS);
  COMPUTE(ATAN);
  COMPUTE(RECIPROCAL);
  COMPUTE(LOG1P);
  COMPUTE(TANH);
  COMPUTE(SIGMOID);
  COMPUTE(EXPM1);
  COMPUTE(ACOSH);
  COMPUTE(ATANH);
  COMPUTE(SIGN);
  COMPUTE(COSH);
  COMPUTE(ROUND);
  COMPUTE(SINH);
  COMPUTE(ASINH);
  COMPUTE(HARDSWISH);

    //case CudaUnaryOpOperation_BNLL:
    //case CudaUnaryOpOperation_ERF:
    //case CudaUnaryOpOperation_ERFC:
    //case CudaUnaryOpOperation_ERFINV:
  return;
}


UnaryExecution::UnaryExecution(UnaryOpOperation opType, Backend* backend) : Execution(backend) {
    auto cudaBackend = static_cast<CUDABackend*>(backend);
    mRuntime      = cudaBackend->getCUDARuntime();
    mOpType = opType;
}
ErrorCode UnaryExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto shape = inputs[0]->shape();
    mCount = CUDABackend::realSize(inputs[0]);
    return NO_ERROR;
}

ErrorCode UnaryExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start UnaryExecution onExecute...");
#endif
    auto type = inputs[0]->getType();
    callUnary((void*)inputs[0]->deviceId(), (void*)outputs[0]->deviceId(), mCount, mRuntime, type, mOpType);
#ifdef LOG_VERBOSE
    MNN_PRINT("end UnaryExecution onExecute...");
#endif
    return NO_ERROR;
}


__global__ void RELU(const float *input, float *output, size_t count, float slope) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float x = input[i];
    float y = x > 0 ? x : x * slope;
    output[i] = y;
  }
  return;
}

class ReluExecution : public Execution {
public:
    ReluExecution(Backend* bn, float slope) : Execution(bn) {
        mSlope = slope;
    }
    virtual ~ReluExecution() = default;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
        auto count = CUDABackend::realSize(inputs[0]);
        int block_num = runtime->blocks_num(count);
        int threads_num = runtime->threads_num();
        auto input = inputs[0]->deviceId();
        auto output = outputs[0]->deviceId();
        RELU<<<block_num, threads_num>>>((float*)input, (float*)output, count, mSlope);
        return NO_ERROR;
    }
private:
    float mSlope;
};


__global__ void CLAMP(const float *input, float *output, size_t count, float minV, float maxV) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float x = input[i];
    float y = min(max(x, minV), maxV);
    output[i] = y;
  }
  return;
}
class Relu6Execution : public Execution {
public:
    Relu6Execution(Backend* bn, float minV, float maxV) : Execution(bn) {
        mMinV = minV;
        mMaxV = maxV;
    }
    virtual ~Relu6Execution() = default;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
        auto count = CUDABackend::realSize(inputs[0]);
        int block_num = runtime->blocks_num(count);
        int threads_num = runtime->threads_num();
        auto input = inputs[0]->deviceId();
        auto output = outputs[0]->deviceId();
        CLAMP<<<block_num, threads_num>>>((float*)input, (float*)output, count, mMinV, mMaxV);
        return NO_ERROR;
    }
private:
    float mMinV;
    float mMaxV;
};

template <typename T1, typename T2>
__global__ void CAST(T1 *input, T2 *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = (T2)(input[i]);
  }
  return;
}

__global__ void CASTBOOL(int32_t *input, int32_t *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i] > 0 ? 1 : 0;
  }
  return;
}

static DataType _mapDataType(DataType src) {
    if (DataType_DT_BOOL == src) {
        return DataType_DT_INT32;
    }
    if (DataType_DT_INT64 == src) {
        return DataType_DT_INT32;
    }
    if (DataType_DT_DOUBLE == src) {
        return DataType_DT_FLOAT;
    }
    return src;
}
class CastExecution : public Execution {
public:
    CastExecution(Backend* bn, DataType dstType) : Execution(bn) {
        mDst = dstType;
    }
    virtual ~CastExecution() = default;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
        auto count = CUDABackend::realSize(inputs[0]);
        int block_num = runtime->blocks_num(count);
        int threads_num = runtime->threads_num();
        auto input = inputs[0]->deviceId();
        auto output = outputs[0]->deviceId();
        auto dstT = _mapDataType(mDst);

        const auto &inputDataType = inputs[0]->getType();

        if (inputDataType.bytes() == 4 && mDst == MNN::DataType_DT_BOOL) {
            CASTBOOL<<<block_num, threads_num>>>((int32_t*)input, (int32_t*)output, count);
        } else if (inputs[0]->buffer().type == outputs[0]->buffer().type) {
            runtime->memcpy((void*)output, (void*)input, count * inputDataType.bytes(), MNNMemcpyDeviceToDevice, true);
        } else if (dstT == MNN::DataType_DT_INT32 && halide_type_of<float>() == inputDataType) {
            CAST<<<block_num, threads_num>>>((float*)input, (int*)output, count);
        } else if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<int32_t>() == inputDataType) {
            CAST<<<block_num, threads_num>>>((int*)input, (float*)output, count);
        } else if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<uint8_t>() == inputDataType) {
            CAST<<<block_num, threads_num>>>((uint8_t*)input, (float*)output, count);
        } else if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<int8_t>() == inputDataType) {
            CAST<<<block_num, threads_num>>>((int8_t*)input, (float*)output, count);
        } else if (dstT == MNN::DataType_DT_INT8 && halide_type_of<float>() == inputDataType) {
            CAST<<<block_num, threads_num>>>((float*)input, (int8_t*)output, count);
        } else if (dstT == MNN::DataType_DT_UINT8 && halide_type_of<float>() == inputDataType) {
            CAST<<<block_num, threads_num>>>((float*)input, (uint8_t*)output, count);
        } else if (dstT == MNN::DataType_DT_UINT8 && halide_type_of<int32_t>() == inputDataType) {
            CAST<<<block_num, threads_num>>>((int32_t*)input, (uint8_t*)output, count);
        } else if (dstT == MNN::DataType_DT_INT32 && halide_type_of<uint8_t>() == inputDataType) {
            CAST<<<block_num, threads_num>>>((uint8_t*)input, (int32_t*)output, count);
        } else if (dstT == MNN::DataType_DT_INT32 && halide_type_of<int8_t>() == inputDataType) {
            CAST<<<block_num, threads_num>>>((int8_t*)input, (int32_t*)output, count);
        }
        return NO_ERROR;
    }
private:
    DataType mDst;
};


class UnaryCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (op->type() == OpType_UnaryOp) {
            switch (op->main_as_UnaryOp()->opType()) {
                // Dont' support erf function
                case UnaryOpOperation_ERF:
                case UnaryOpOperation_ERFC:
                case UnaryOpOperation_ERFINV:
                    return nullptr;
                default:
                    return new UnaryExecution(op->main_as_UnaryOp()->opType(), backend);
            }
        }
        if (op->type() == OpType_Sigmoid) {
            return new UnaryExecution(UnaryOpOperation_SIGMOID, backend);
        }
        if (op->type() == OpType_TanH) {
            return new UnaryExecution(UnaryOpOperation_TANH, backend);
        }
        if (op->type() == OpType_ReLU) {
            float slope = 0.0f;
            if (nullptr != op->main_as_Relu()) {
                slope = op->main_as_Relu()->slope();
            }
            return new ReluExecution(backend, slope);
        }
        if (op->type() == OpType_ReLU6) {
            float minV = 0.0f;
            float maxV = 6.0f;
            if (nullptr != op->main()) {
                auto p = op->main_as_Relu6();
                minV = p->minValue();
                maxV = p->maxValue();
            }
            return new Relu6Execution(backend, minV, maxV);
        }
        if (op->type() == OpType_Cast) {
            return new CastExecution(backend, op->main_as_CastParam()->dstT());
        }
        return nullptr;
    }
};

CUDACreatorRegister<UnaryCreator> __UnaryExecution(OpType_UnaryOp);
CUDACreatorRegister<UnaryCreator> __SigmoidExecution(OpType_Sigmoid);
CUDACreatorRegister<UnaryCreator> __TanhExecution(OpType_TanH);
CUDACreatorRegister<UnaryCreator> __ReluExecution(OpType_ReLU);
CUDACreatorRegister<UnaryCreator> __Relu6Execution(OpType_ReLU6);
CUDACreatorRegister<UnaryCreator> __CastExecution(OpType_Cast);
} // namespace CUDA
} // namespace MNN
