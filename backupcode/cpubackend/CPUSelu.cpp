//
//  CPUSelu.cpp
//  MNN
//
//  Created by MNN on 2018/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUSelu.hpp"
#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"

namespace MNN {

CPUSelu::CPUSelu(Backend *b, const MNN::Op *op) : MNN::Execution(b) {
    auto selu = op->main_as_Selu();
    mScale    = selu->scale();
    mAlpha    = selu->alpha();
}

ErrorCode CPUSelu::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(1 == outputs.size());
    MNN_ASSERT(inputs[0]->buffer().type.bytes() == 4);
    auto scaleAlpha = mAlpha * mScale;
    auto ptr        = inputs[0]->host<float>();
    auto outptr     = outputs[0]->host<float>();
    int size        = inputs[0]->size() / sizeof(float);
    for (int i = 0; i < size; i++) {
        if (ptr[i] < 0.f) {
            outptr[i] = scaleAlpha * (::expf(ptr[i]) - 1.f);
        } else {
            outptr[i] = mScale * ptr[i];
        }
    }

    return NO_ERROR;
}

class CPUSeluCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUSelu(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUSeluCreator, OpType_Selu);

} // namespace MNN
