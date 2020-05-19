//
//  CPUFill.cpp
//  MNN
//
//  Created by MNN on 2018/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUFill.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"

namespace MNN {

CPUFill::CPUFill(Backend *backend) : Execution(backend) {
    // nothing to do
}

ErrorCode CPUFill::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    int bytes = outputs[0]->getType().bytes();
    int size  = outputs[0]->elementSize();

    uint8_t *value  = inputs[1]->host<uint8_t>();
    uint8_t *buffer = outputs[0]->host<uint8_t>();
    for (int i = 0; i < size; ++i, buffer += bytes) {
        memcpy(buffer, value, bytes);
    }

    return NO_ERROR;
}

class CPUFillCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUFill(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUFillCreator, OpType_Fill);
} // namespace MNN
