//
//  CPUGatherV2.cpp
//  MNN
//
//  Created by MNN on 2018/08/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUGatherV2.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"

namespace MNN {

template <typename T>
CPUGatherV2<T>::CPUGatherV2(Backend *b, const MNN::Op *op) : MNN::Execution(b), mOp(op) {
    // nothing to do
}

template <typename T>
ErrorCode CPUGatherV2<T>::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

template <typename T>
ErrorCode CPUGatherV2<T>::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto params  = inputs[0];
    auto indices = inputs[1];
    auto output  = outputs[0];
    int axis     = 0;
    if (inputs.size() == 3) {
        const Tensor *axisTensor = inputs[2];
        axis                     = axisTensor->host<int32_t>()[0];
    }
    MNN_ASSERT(axis > -params->buffer().dimensions && axis < params->buffer().dimensions);

    if (axis < 0) {
        axis = params->buffer().dimensions + axis;
    }
    const int gatherDimSize = params->buffer().dim[axis].extent;
    const int N             = indices->elementSize();
    MNN_ASSERT(gatherDimSize <= std::numeric_limits<int32_t>::max());

    // TODO : CURRUNT ONLY SUPPORT AXIS == 0
    MNN_ASSERT(0 == axis);
    const int limit          = params->length(0);
    const int firstDimStride = params->buffer().dim[0].stride;
    const int *indicesPtr    = indices->host<int32_t>();
    const auto inputPtr      = params->host<T>();
    auto outputPtr           = output->host<T>();
    for (int i = 0; i < N; i++) {
        if (indicesPtr[i] < 0 || indicesPtr[i] > limit) {
            return INPUT_DATA_ERROR;
        }
        memcpy(outputPtr + i * firstDimStride, inputPtr + firstDimStride * indicesPtr[i], sizeof(T) * firstDimStride);
    }

    return NO_ERROR;
}

class CPUGatherV2Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        switch (inputs[0]->getType().code) {
            case halide_type_int:
                return new CPUGatherV2<int32_t>(backend, op);
            case halide_type_float:
                return new CPUGatherV2<float>(backend, op);
            default:
                MNN_ASSERT(false); // unsupported type
                return nullptr;
        }
    }
};

REGISTER_CPU_OP_CREATOR(CPUGatherV2Creator, OpType_GatherV2);

} // namespace MNN
