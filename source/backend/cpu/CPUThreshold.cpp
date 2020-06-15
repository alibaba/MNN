//
//  CPUThreshold.cpp
//  MNN
//
//  Created by MNN on 2019/12/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUThreshold.hpp"
#include "backend/cpu/CPUBackend.hpp"

namespace MNN {
ErrorCode CPUThreshold::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];

    const float* srcData = input->host<float>();
    float* dstData = output->host<float>();
    const int size = input->elementSize();

    for (int i = 0; i < size; ++i) {
        if (srcData[i] > mThreshold) {
            dstData[i] = 1.0f;
        } else {
            dstData[i] = 0.0f;
        }
    }

    return NO_ERROR;
}

class CPUThresholdCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPUThreshold(backend, op->main_as_ELU()->alpha());
    }
};

REGISTER_CPU_OP_CREATOR(CPUThresholdCreator, OpType_Threshold);
} // namespace MNN
