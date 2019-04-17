//
//  CPUQuantizedReshape.cpp
//  MNN
//
//  Created by MNN on 2018/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUQuantizedReshape.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"

namespace MNN {

CPUQuantizedReshape::CPUQuantizedReshape(const MNN::Op *op, Backend *b) : MNN::Execution(b) {
    auto param = op->main_as_QuantizedReshape();
    mIstflite  = param->modelFormat() == MNN::ModeFormat_TFLITE;
}

ErrorCode CPUQuantizedReshape::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

ErrorCode CPUQuantizedReshape::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(3 == inputs.size() || 4 == inputs.size() || 1 == inputs.size());
    MNN_ASSERT(3 == outputs.size() || inputs.size() == 1);

    auto &input  = inputs[0]->buffer();
    auto &output = outputs[0]->buffer();

    MNN_ASSERT(input.dimensions <= 1 || input.dim[1].flags == 0);

    if (input.dimensions <= 1 || input.dim[1].flags == 0) {
        ::memcpy(output.host, input.host, inputs[0]->size());
    }

    if (mIstflite == false) {
        ((float *)(outputs[1]->buffer().host))[0] = inputs[2]->host<float>()[0];
        ((float *)(outputs[2]->buffer().host))[0] = inputs[3]->host<float>()[0];
    }

    return NO_ERROR;
}

class CPUQuantizedReshapeCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUQuantizedReshape(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUQuantizedReshapeCreator, OpType_QuantizedReshape);

} // namespace MNN
