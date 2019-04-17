//
//  CPUSqueeze.cpp
//  MNN
//
//  Created by MNN on 2018/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUSqueeze.hpp"
#include "CPUBackend.hpp"

namespace MNN {

CPUSqueeze::CPUSqueeze(Backend* b, const SqueezeParam* parameter) : MNN::Execution(b), mParameter(parameter) {
    // nothing to do
}

ErrorCode CPUSqueeze::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto& dstBuffer = outputs[0]->buffer();
    auto& srcBuffer = inputs[0]->buffer();
    memcpy(dstBuffer.host, srcBuffer.host, outputs[0]->size());
    return NO_ERROR;
}

class CPUSqueezeCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUSqueeze(backend, op->main_as_SqueezeParam());
    }
};

REGISTER_CPU_OP_CREATOR(CPUSqueezeCreator, OpType_Squeeze);
} // namespace MNN
