//
//  CPURelu.cpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPURelu.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "CPUBackend.hpp"
#include <string.h>
namespace MNN {
ErrorCode CPURelu::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto& ib = inputs[0]->buffer();
    auto& ob = outputs[0]->buffer();

    const float* srcO = (const float*)ib.host;
    float* dstO       = (float*)ob.host;
    auto size         = inputs[0]->size() / sizeof(float);
    auto numberThread = ((CPUBackend*)backend())->threadNumber();
    auto sizeQuad     = size / 4;
    auto remain       = sizeQuad * 4;
    int sizeDivide = sizeQuad / numberThread;
    if (sizeQuad > 0) {
        MNN_CONCURRENCY_BEGIN(tId, numberThread) {
            int number = sizeDivide;
            if (tId == numberThread - 1) {
                number = sizeQuad - tId * sizeDivide;
            }
            MNNReluWithSlope(dstO + 4 * tId * sizeDivide, srcO + 4 * tId * sizeDivide, number, mSlope);
        }
        MNN_CONCURRENCY_END();
    }
    for (int j = remain; j < size; ++j) {
        if (srcO[j] < 0) {
            dstO[j] = srcO[j] * mSlope;
        } else {
            dstO[j] = srcO[j];
        }
    }
    return NO_ERROR;
}

ErrorCode CPURelu6::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto& ib = inputs[0]->buffer();
    auto& ob = outputs[0]->buffer();

    const float* srcO = (const float*)ib.host;
    float* dstO       = (float*)ob.host;
    auto size         = inputs[0]->elementSize();
    auto numberThread = ((CPUBackend*)backend())->threadNumber();
    auto sizeQuad     = size / 4;
    auto remain       = sizeQuad * 4;
    int sizeDivide = sizeQuad / numberThread;

    std::vector<float> bias = {0.0f, 0.0f, 0.0f, 0.0f};
    MNN_CONCURRENCY_BEGIN(tId, numberThread) {
        int number = sizeDivide;
        if (tId == numberThread - 1) {
            number = sizeQuad - tId * sizeDivide;
        }
        ::memcpy(dstO + tId * sizeDivide * 4, srcO + tId * sizeDivide * 4, number * 4 * sizeof(float));
        MNNAddBiasRelu6(dstO + tId * sizeDivide * 4, bias.data(), number, 1);
    }
    MNN_CONCURRENCY_END();
    MNNRelu6(dstO + remain, srcO + remain, size - remain);
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
    const float* srcO   = (const float*)ib.host;
    float* dstO         = (float*)ob.host;
    int sizeQuad        = width * height;
    auto totalCount = batch * depthQuad;
    auto numberThread = ((CPUBackend*)backend())->threadNumber();
    MNN_CONCURRENCY_BEGIN(tId, numberThread) {
        for (int b=tId; b<totalCount; b+=numberThread) {
            auto c = b % depthQuad;
            MNNReluWithSlopeChannel(dstO + sizeQuad * 4 * b, srcO + sizeQuad * 4 * b, mSlope.get() + 4 * c, sizeQuad, 1);
        }
    }
    MNN_CONCURRENCY_END();
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
