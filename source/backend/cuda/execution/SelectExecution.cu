//
//  SelectExecution.cpp
//  MNN
//
//  Created by MNN on 2021/12/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SelectExecution.hpp"
#include "core/Macro.h"
#include <cuda_runtime.h>

namespace MNN {
namespace CUDA {
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template<typename T>
__global__ void SELECT(const int size, const int* input0, const T* input1, const T* input2,
    int s1, int s2, T* output) {
    CUDA_KERNEL_LOOP(i, size) {
        if (input0[i] > 0) {
            output[i] = input1[i*s1];
        } else {
            output[i] = input2[i*s2];
        }
    }
}

SelectExecution::SelectExecution(Backend* backend) : Execution(backend) {
    // Do nothing
}
ErrorCode SelectExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Do nothing
    return NO_ERROR;
}

ErrorCode SelectExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start SelectExecution onExecute...");
#endif
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto count = CUDABackend::realSize(inputs[0]);
    auto inputS1 = CUDABackend::realSize(inputs[1]);
    auto inputS2 = CUDABackend::realSize(inputs[2]);
    int s1 = inputS1 == 1 ? 0 : 1;
    int s2 = inputS2 == 1 ? 0 : 1;
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        SELECT<<<block_num, threads_num>>>(count, (const int*)(inputs[0]->deviceId()), (const half*)(inputs[1]->deviceId()), (const half*)(inputs[2]->deviceId()), \
                                            s1, s2, (half*)outputs[0]->deviceId());
        checkKernelErrors;
    } else {
        SELECT<<<block_num, threads_num>>>(count, (const int*)(inputs[0]->deviceId()), (const float*)(inputs[1]->deviceId()), (const float*)(inputs[2]->deviceId()), \
                                            s1, s2, (float*)outputs[0]->deviceId());
        checkKernelErrors;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end SelectExecution onExecute...");
#endif
    return NO_ERROR;
}


class SelectCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new SelectExecution(backend);
    }
};

CUDACreatorRegister<SelectCreator> __SelectExecution(OpType_Select);
} // namespace CUDA
} // namespace MNN
