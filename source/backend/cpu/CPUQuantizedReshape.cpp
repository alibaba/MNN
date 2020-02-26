//
//  CPUQuantizedReshape.cpp
//  MNN
//
//  Created by MNN on 2018/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TFLITE_QUAN
#include "backend/cpu/CPUQuantizedReshape.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"

namespace MNN {

CPUQuantizedReshape::CPUQuantizedReshape(Backend *b) : MNN::Execution(b) {
    // Do nothing
}

ErrorCode CPUQuantizedReshape::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

ErrorCode CPUQuantizedReshape::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(3 == inputs.size() || 4 == inputs.size() || 1 == inputs.size());
    MNN_ASSERT(3 == outputs.size() || inputs.size() == 1);

    auto &input  = inputs[0]->buffer();
    auto &output = outputs[0]->buffer();
    ::memcpy(output.host, input.host, inputs[0]->size());

    return NO_ERROR;
}

class CPUQuantizedReshapeCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUQuantizedReshape(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUQuantizedReshapeCreator, OpType_QuantizedReshape);

} // namespace MNN
#endif
