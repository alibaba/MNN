//
//  CPUReluGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUReluGrad.hpp"
#include "core/Concurrency.h"
#include "CPUBackend.hpp"
namespace MNN {
ErrorCode CPUReluGrad::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(0 == mSlope);
    auto reluOrigin = inputs[0];
    auto reluDiff   = inputs[1];
    auto outputDiff = outputs[0];
    auto size       = outputDiff->elementSize();

    auto reluOriginPtr = reluOrigin->host<float>();
    auto reluDiffPtr   = reluDiff->host<float>();
    auto outputDiffPtr = outputDiff->host<float>();
    auto numberThread = ((CPUBackend*)backend())->threadNumber();
    MNN_CONCURRENCY_BEGIN(tId, numberThread) {
        for (int n = tId; n < size; n+=numberThread) {
            if (reluOriginPtr[n] > 0.0f) {
                outputDiffPtr[n] = reluDiffPtr[n];
            } else {
                outputDiffPtr[n] = 0.0f;
            }
        }
    }
    MNN_CONCURRENCY_END();

    return NO_ERROR;
}
class CPURelu6Grad : public Execution {
public:
    CPURelu6Grad(Backend *bn) : Execution(bn) {
        //Do nothing
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto reluOrigin = inputs[0];
        auto reluDiff   = inputs[1];
        auto outputDiff = outputs[0];
        auto size       = outputDiff->elementSize();

        auto reluOriginPtr = reluOrigin->host<float>();
        auto reluDiffPtr   = reluDiff->host<float>();
        auto outputDiffPtr = outputDiff->host<float>();
        auto numberThread = ((CPUBackend*)backend())->threadNumber();
        MNN_CONCURRENCY_BEGIN(tId, numberThread) {
            for (int n = tId; n < size; n+=numberThread) {
                if (reluOriginPtr[n] > 0.0f && reluOriginPtr[n] <= 6.0f) {
                    outputDiffPtr[n] = reluDiffPtr[n];
                } else {
                    outputDiffPtr[n] = 0.0f;
                }
            }
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }
};
class CPUReluGradCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (op->type() == OpType_ReluGrad) {
            auto slope = op->main_as_Relu()->slope();
            return new CPUReluGrad(slope, backend);
        }
        if (op->type() == OpType_Relu6Grad) {
            return new CPURelu6Grad(backend);
        }
        return nullptr;
    }
};

REGISTER_CPU_OP_CREATOR(CPUReluGradCreator, OpType_ReluGrad);
REGISTER_CPU_OP_CREATOR(CPUReluGradCreator, OpType_Relu6Grad);
} // namespace MNN
