//
//  CPURank.cpp
//  MNN
//
//  Created by MNN on 2018/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPURank.hpp"
#include "CPUBackend.hpp"

namespace MNN {

CPURank::CPURank(Backend *backend) : Execution(backend) {
    // nothing to do
}

ErrorCode CPURank::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    outputs[0]->host<int32_t>()[0] = inputs[0]->buffer().dimensions;
    return NO_ERROR;
}

class CPURankCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPURank(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPURankCreator, OpType_Rank);
} // namespace MNN
