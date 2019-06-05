//
//  CPURelu.cpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPURelu.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"

namespace MNN {
ErrorCode CPURelu::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto& ib = inputs[0]->buffer();
    auto& ob = outputs[0]->buffer();

    const float* srcO = (const float*)ib.host;
    float* dstO       = (float*)ob.host;
    auto size         = inputs[0]->size() / sizeof(float);
    auto sizeQuad     = size / 4;
    auto remain       = size - sizeQuad * 4;

    MNNReluWithSlope(dstO, srcO, sizeQuad, mSlope);

    if (remain > 0) {
        MNNReluWithSlope(dstO + size - 4, srcO + size - 4, 1, mSlope);
    }

    return NO_ERROR;
}

ErrorCode CPURelu6::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto& ib = inputs[0]->buffer();
    auto& ob = outputs[0]->buffer();

    const float* srcO = (const float*)ib.host;
    float* dstO       = (float*)ob.host;
    auto size         = inputs[0]->size() / sizeof(float);

    MNNRelu6(dstO, srcO, size);
    return NO_ERROR;
}

CPUPRelu::CPUPRelu(Backend* b, const Op* op) : MNN::Execution(b) {
    auto c = op->main_as_PRelu();
    mSlope.reset(ALIGN_UP4(c->slopeCount()));
    mSlope.clear();
    ::memcpy(mSlope.get(), c->slope()->data(), c->slopeCount() * sizeof(float));
}

ErrorCode CPUPRelu::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto& ib            = inputs[0]->buffer();
    auto& ob            = outputs[0]->buffer();
    const int width     = ib.dim[3].extent;
    const int height    = ib.dim[2].extent;
    const int channel   = ib.dim[1].extent;
    const int batch     = ib.dim[0].extent;
    const int depthQuad = UP_DIV(channel, 4);
    const int batchSize = depthQuad * 4 * width * height;
    const float* srcO   = (const float*)ib.host;
    float* dstO         = (float*)ob.host;
    int sizeQuad        = width * height;

    for (int b = 0; b < batch; ++b) {
        auto src = srcO + b * batchSize;
        auto dst = dstO + b * batchSize;
        MNNReluWithSlopeChannel(dst, src, mSlope.get(), sizeQuad, depthQuad);
    }
    return NO_ERROR;
}

class CPUReluCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        if (op->type() == OpType_ReLU) {
            auto slope = 0.0f;
            if (nullptr != op->main() && OpParameter_Relu == op->main_type()) {
                slope = op->main_as_Relu()->slope();
            }
            return new CPURelu(backend, slope);
        }
        MNN_ASSERT(op->type() == OpType_PReLU);
        if (op->main_as_PRelu()->slopeCount() == 1) {
            return new CPURelu(backend, op->main_as_PRelu()->slope()->data()[0]);
        }
        return new CPUPRelu(backend, op);
    }
};

class CPURelu6Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPURelu6(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUReluCreator, OpType_ReLU);
REGISTER_CPU_OP_CREATOR(CPUReluCreator, OpType_PReLU);
REGISTER_CPU_OP_CREATOR(CPURelu6Creator, OpType_ReLU6);
} // namespace MNN
