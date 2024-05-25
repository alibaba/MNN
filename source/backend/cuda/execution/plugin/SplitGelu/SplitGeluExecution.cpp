//
//  SplitGeluExecution.cpp
//  MNN
//
//  Created by MNN on 2023/09/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "SplitGeluExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace CUDA {

SplitGeluExecution::SplitGeluExecution(Backend* backend) : Execution(backend) {
    mFDiv = 1.4140625F;
    mFAdd = 1.F;
    mFMul = 0.5F;
}
ErrorCode SplitGeluExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Nothing todo
    return NO_ERROR;
}

ErrorCode SplitGeluExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start SplitGeluExecution onExecute...");
#endif
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();

    int32_t const gridSize = inputs[0]->length(0) * inputs[0]->length(1);
    int32_t const nHalfHiddenSize = inputs[0]->length(2) / 2; // HHS
    if(inputs.size() > 1) {
        MNN_ASSERT(inputs[1]->dimensions() == 1);
        MNN_ASSERT(inputs[1]->length(0) == inputs[0]->length(2));
    }

    if(static_cast<CUDABackend*>(backend())->useFp16()) {
        auto const input0 = static_cast<half const*>((void *)inputs[0]->deviceId());
        void* input1 = nullptr;
        if(inputs.size() > 1) {
            input1 = (void *)inputs[1]->deviceId();
        } 
        auto output = static_cast<half*>((void *)outputs[0]->deviceId());
        launchSplitGeLUKernel<half>(gridSize, nHalfHiddenSize, input0, static_cast<half const*>(input1), output, mFDiv, mFAdd, mFMul);
    } else {
        auto const input0 = static_cast<float const*>((void *)inputs[0]->deviceId());
        void* input1 = nullptr;
        if(inputs.size() > 1) {
            input1 = (void *)inputs[1]->deviceId();
        }
        auto output = static_cast<float*>((void *)outputs[0]->deviceId());
        launchSplitGeLUKernel<float>(gridSize, nHalfHiddenSize, input0, static_cast<float const*>(input1), output, mFDiv, mFAdd, mFMul);
    }
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end SplitGeluExecution onExecute...");
#endif
    return NO_ERROR;
}


class SplitGeluCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new SplitGeluExecution(backend);
    }
};

CUDACreatorRegister<SplitGeluCreator> __SplitGeluExecution(OpType_SplitGeLU);
} // namespace CUDA
} // namespace MNN
#endif