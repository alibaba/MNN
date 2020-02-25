//
//  CPUGather.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUGather.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"

namespace MNN {

CPUGather::CPUGather(Backend *b, const MNN::Op *op) : MNN::Execution(b), mOp(op) {
    // nothing to do
}

ErrorCode CPUGather::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

ErrorCode CPUGather::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto embedding = inputs[0];
    auto indices   = inputs[1];
    auto output    = outputs[0];

    auto bytes = embedding->buffer().type.bytes();

    const size_t indicesCount = indices->elementSize();
    const auto limit          = embedding->length(0);

    auto outputData          = output->host<uint8_t>();
    const auto *inputData   = embedding->host<uint8_t>();
    const int firstDimStride = embedding->buffer().dim[0].stride * bytes;
    const int *indicesData   = indices->host<int32_t>();

    for (int i = 0; i < indicesCount; i++) {
        if (indicesData[i] < 0 || indicesData[i] > limit) {
            return INPUT_DATA_ERROR;
        }
        memcpy(outputData + i * firstDimStride, inputData + firstDimStride * indicesData[i],
               firstDimStride);
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
