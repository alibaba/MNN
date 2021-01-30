//
//  CPUSelect.cpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#include "backend/cpu/CPUSelect.hpp"
namespace MNN {

ErrorCode CPUSelect::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inSize1 = inputs[1]->elementSize();
    auto inSize2 = inputs[2]->elementSize();
    auto outSize = outputs[0]->elementSize();
    MNN_ASSERT(inputs[0]->elementSize() == outSize);
    MNN_ASSERT(inSize1 == 1 || inSize1 == outSize);
    MNN_ASSERT(inSize2 == 1 || inSize2 == outSize);
    auto output = outputs[0]->host<float>();
    auto select = inputs[0]->host<int32_t>();
    auto input0 = inputs[1]->host<float>();
    auto input1 = inputs[2]->host<float>();
    for (int i = 0; i < outSize; i++) {
        if (select[i]) {
            if (inSize1 == 1) {
                output[i] = input0[0];
            } else {
                output[i] = input0[i];
            }
        } else {
            if (inSize2 == 1) {
                output[i] = input1[0];
            } else {
                output[i] = input1[i];
            }
        }
    }
    return NO_ERROR;
}

class CPUSelectCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUSelect(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUSelectCreator, OpType_Select);
} // namespace MNN
