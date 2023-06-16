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
#include "Raster.cuh"
#include "backend/cuda/core/CUDABackend.hpp"
#include <cuda_runtime.h>

namespace MNN {
namespace CUDA {

void callUnary(void *input, void *output, size_t count, MNN::CUDARuntime* runtime, halide_type_t data_type,
   MNN::UnaryOpOperation op_type)
{
    Tensor::InsideDescribe::Region reg;
    reg.size[2] = count;
    UnaryBlit((uint8_t*)output, (const uint8_t*)input, reg.size, reg.src.stride, reg.dst.stride, data_type.bytes(), runtime, op_type);
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
    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        type.bits = 16;
    }
    //MNN_PRINT("unary size:%d\n", mCount);
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

__global__ void RELU_Half(const half *input, half *output, size_t count, float slope) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float x = input[i];
    float y = x > 0 ? x : x * slope;
    output[i] = (half)y;
  }
  return;
}

__global__ void RELU_INT8(const int8_t *input, int8_t *output, size_t count, int8_t zeroPoint) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
      int8_t x = input[i];
      int8_t y = x > zeroPoint ? x : zeroPoint;
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
        if (TensorUtils::getDescribe(outputs[0])->quantAttr != nullptr && TensorUtils::getDescribe(outputs[0])->type == DataType_DT_INT8) {
            auto inInfo = TensorUtils::getQuantInfo(inputs[0]);
            auto outInfo = TensorUtils::getQuantInfo(outputs[0]);
            if (inInfo != outInfo) {
                MNN_PRINT("this relu int8 implementation has error when input output quant info mismatch\n");
            }
            if(mSlope > 0.0f || mSlope < 0.0f) {
                MNN_PRINT("Warning, CUDA only support Relu int8, PReLU int8 not support yet!\n");
            }
            int8_t zeroPoint = int8_t(outInfo[1]);
            RELU_INT8<<<block_num, threads_num>>>((const int8_t*)input, (int8_t*)output, count, zeroPoint);
            checkKernelErrors;
            return NO_ERROR;
        }

        if (static_cast<CUDABackend*>(backend())->useFp16()) {
            RELU_Half<<<block_num, threads_num>>>((half*)input, (half*)output, count, mSlope);
            checkKernelErrors;
        } else {
            RELU<<<block_num, threads_num>>>((float*)input, (float*)output, count, mSlope);
            checkKernelErrors;
        }
        return NO_ERROR;
    }
private:
    float mSlope;
};


template<typename T>
__global__ void CLAMP(const T *input, T *output, size_t count, float minV, float maxV) {
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
        if (static_cast<CUDABackend*>(backend())->useFp16()) {
            CLAMP<<<block_num, threads_num>>>((half*)input, (half*)output, count, mMinV, mMaxV);
        } else {
            CLAMP<<<block_num, threads_num>>>((float*)input, (float*)output, count, mMinV, mMaxV);
        }
        return NO_ERROR;
    }
private:
    float mMinV;
    float mMaxV;
};

class UnaryCreator : public CUDABackend::Creator {
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
        return nullptr;
    }
};

CUDACreatorRegister<UnaryCreator> __UnaryExecution(OpType_UnaryOp);
CUDACreatorRegister<UnaryCreator> __SigmoidExecution(OpType_Sigmoid);
CUDACreatorRegister<UnaryCreator> __TanhExecution(OpType_TanH);
CUDACreatorRegister<UnaryCreator> __ReluExecution(OpType_ReLU);
CUDACreatorRegister<UnaryCreator> __Relu6Execution(OpType_ReLU6);
} // namespace CUDA
} // namespace MNN
