//
//  CPUSelect.cpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#include "CPUSelect.hpp"
namespace MNN {
ErrorCode CPUSelect::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto select    = inputs[0];
    auto outputPtr = outputs[0]->host<float>();
    auto input0Ptr = inputs[1]->host<float>();
    auto input1Ptr = inputs[2]->host<float>();
    auto selectPtr = select->host<int32_t>();
    auto size      = select->elementSize();
    for (int i = 0; i < size; ++i) {
        if (selectPtr[i] > 0) {
            outputPtr[i] = input0Ptr[i];
        } else {
            outputPtr[i] = input1Ptr[i];
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
