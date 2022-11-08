//
//  RangeExecution.cpp
//  MNN
//
//  Created by MNN on 2022/04/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "RangeExecution.hpp"
#include "core/Macro.h"
#include <cuda_runtime.h>

namespace MNN {
namespace CUDA {
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template<typename T>
__global__ void RANGE(const int size, const T* input0, const T* input2, T* output) {
    CUDA_KERNEL_LOOP(i, size) {
        T start = input0[0];
        T step = input2[0];
        output[i] = start + (T)i * step;
    }
}

RangeExecution::RangeExecution(Backend* backend) : Execution(backend) {
    // Do nothing
}
ErrorCode RangeExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Do nothing
    return NO_ERROR;
}

ErrorCode RangeExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start RangeExecution onExecute...");
#endif

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto count = outputs[0]->buffer().dim[0].extent;

    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();

    auto code = inputs[0]->getType().code;
    if(code == halide_type_int) {
        RANGE<<<block_num, threads_num>>>(count, (const int*)(inputs[0]->deviceId()), (const int*)(inputs[2]->deviceId()), (int*)outputs[0]->deviceId());
    } else if (static_cast<CUDABackend*>(backend())->useFp16()) {
        RANGE<<<block_num, threads_num>>>(count, (const half*)(inputs[0]->deviceId()), (const half*)(inputs[2]->deviceId()), (half*)outputs[0]->deviceId());
    } else {
        RANGE<<<block_num, threads_num>>>(count, (const float*)(inputs[0]->deviceId()), (const float*)(inputs[2]->deviceId()), (float*)outputs[0]->deviceId());
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end RangeExecution onExecute...");
#endif
    return NO_ERROR;
}


class RangeCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto code = inputs[0]->getType().code;
        if(code == halide_type_int) {
            return new RangeExecution(backend);
        } else if(code == halide_type_float) {
            return new RangeExecution(backend);
        } else {
            MNN_PRINT("MNN CUDA only support Range datatype float or int");
            return nullptr;
        }
    }
};

CUDACreatorRegister<RangeCreator> __RangeExecution(OpType_Range);
} // namespace CUDA
} // namespace MNN
