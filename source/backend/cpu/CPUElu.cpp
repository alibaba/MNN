//
//  CPUElu.cpp
//  MNN
//
//  Created by MNN on 2019/09/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cmath>
#include "backend/cpu/CPUElu.hpp"
#include "backend/cpu/CPUBackend.hpp"

namespace MNN {
ErrorCode CPUElu::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];

    const float* srcData = input->host<float>();
    float* dstData = output->host<float>();
    const int size = input->elementSize();

    for (int i = 0; i < size; ++i) {
        if (srcData[i] >= 0) {
            dstData[i] = srcData[i];
        } else {
            dstData[i] = mAlpha * (expf(srcData[i]) - 1);
        }
    }

    return NO_ERROR;
}

class CPUEluCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPUElu(backend, op->main_as_ELU()->alpha());
    }
};

REGISTER_CPU_OP_CREATOR(CPUEluCreator, OpType_ELU);
} // namespace MNN
