//
//  CPUGatherV2.cpp
//  MNN
//
//  Created by MNN on 2018/08/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUGatherV2.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"

namespace MNN {

CPUGatherV2::CPUGatherV2(Backend *b, const Op* op) : MNN::Execution(b), mOp(op) {
    // nothing to do
}

ErrorCode CPUGatherV2::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto params  = inputs[0];
    mAxis = 0;
    if (inputs.size() == 3) {
        const Tensor *axisTensor = inputs[2];
        mAxis                     = axisTensor->host<int32_t>()[0];
    }
    if (mOp->main_type() == OpParameter_Axis) {
        mAxis = mOp->main_as_Axis()->axis();
    }
    MNN_ASSERT(mAxis > -params->buffer().dimensions && mAxis < params->buffer().dimensions);

    if (mAxis < 0) {
        mAxis = params->buffer().dimensions + mAxis;
    }
    return NO_ERROR;
}

ErrorCode CPUGatherV2::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto params  = inputs[0];
    auto indices = inputs[1];
    auto output  = outputs[0];
    int axis     = mAxis;
    const int N             = indices->elementSize();
    int inside = 1;
    int outside = 1;
    for (int i=0; i<axis; ++i) {
        outside *= params->length(i);
    }
    for (int i=axis+1; i<params->dimensions(); ++i) {
        inside *= params->length(i);
    }
    const int limit          = params->length(axis);
    auto bytes = output->buffer().type.bytes();
    const int insideStride = inside * bytes;
    const int outputOutsideStride = inside * N * bytes;
    const int inputOutsideStride = inside * bytes *inputs[0]->length(axis);
    const int *indicesPtr    = indices->host<int32_t>();
    const auto inputPtr      = params->host<uint8_t>();
    auto outputPtr           = output->host<uint8_t>();
    for (int o=0; o<outside; ++o) {
        auto outputO = outputPtr + outputOutsideStride * o;
        auto inputO = inputPtr + inputOutsideStride * o;
        for (int i = 0; i < N; i++) {
            if (indicesPtr[i] < 0 || indicesPtr[i] > limit) {
                ::memset(outputO + i * insideStride, 0, insideStride);
                continue;
            }
            memcpy(outputO + i * insideStride, inputO + insideStride * indicesPtr[i], insideStride);
        }
    }
    return NO_ERROR;
}

class CPUGatherV2Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUGatherV2(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUGatherV2Creator, OpType_GatherV2);
REGISTER_CPU_OP_CREATOR(CPUGatherV2Creator, OpType_Gather);

} // namespace MNN
