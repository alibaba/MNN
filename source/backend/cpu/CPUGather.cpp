//
//  CPUGather.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUGather.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"

namespace MNN {

CPUGather::CPUGather(Backend *b, const MNN::Op *op) : MNN::Execution(b), mOp(op) {
    // nothing to do
}

ErrorCode CPUGather::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == outputs.size());
    auto indices = inputs[1];
    if(indices->buffer().type.bits != 32) {
        return NOT_SUPPORT;
    }
    return NO_ERROR;
}

ErrorCode CPUGather::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto embedding = inputs[0];
    auto indices   = inputs[1];
    auto output    = outputs[0];

    MNN_ASSERT(embedding->buffer().type.bits == 32);

    const size_t indicesCount = indices->elementSize();
    const auto limit          = embedding->length(0);

    auto outputData          = output->host<float>();
    const float *inputData   = embedding->host<float>();
    const int firstDimStride = embedding->buffer().dim[0].stride;
    const int *indicesData   = indices->host<int32_t>();

    for (int i = 0; i < indicesCount; i++) {
        if (indicesData[i] < 0 || indicesData[i] > limit) {
            return INPUT_DATA_ERROR;
        }
        memcpy(outputData + i * firstDimStride, inputData + firstDimStride * indicesData[i],
               sizeof(float) * firstDimStride);
    }

    return NO_ERROR;
}

class CPUGatherCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUGather(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUGatherCreator, OpType_Gather);

} // namespace MNN
