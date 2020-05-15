//
//  CPUSelect.cpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#include "backend/cpu/CPUSelect.hpp"
namespace MNN {

static void ApplySelectAtAxis(int axis, int dimensions, const int* select,
                              const float* input0, const float* input1, float* output,
                              const std::vector<int>& select_shape,
                              const std::vector<int>& input_shape,
                              const std::vector<int>& select_stride,
                              const std::vector<int>& input_stride) {
    if (axis >= dimensions) {
        if (*select) {
            *output = *input0;
        } else {
            *output = *input1;
        }
        return;
    }

    for (int i = 0; i < input_shape[axis]; ++i) {
        ApplySelectAtAxis(axis + 1, dimensions, select, input0, input1,
                          output, select_shape, input_shape, select_stride, input_stride);
        input0 += input_stride[axis];
        input1 += input_stride[axis];
        output += input_stride[axis];
        if (select_shape[axis] == input_shape[axis]) {
            select += select_stride[axis];
        }
    }
}

ErrorCode CPUSelect::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    int dimensions = inputs[1]->dimensions();
    auto select    = inputs[0];
    MNN_ASSERT(inputs.size() == 3);
    MNN_ASSERT(dimensions == inputs[2]->dimensions());
    for (int i = 0; i < dimensions; ++i) {
        MNN_ASSERT(inputs[1]->length(i) == inputs[2]->length(i));
    }

    MNN_ASSERT(select->dimensions() <= dimensions);
    for (int i = 0; i < select->dimensions(); ++i) {
        MNN_ASSERT(select->length(i) == 1 ||
                   select->length(i) == inputs[1]->length(i));
    }

    auto outputPtr = outputs[0]->host<float>();
    auto input0Ptr = inputs[1]->host<float>();
    auto input1Ptr = inputs[2]->host<float>();
    auto selectPtr = select->host<int32_t>();

    std::vector<int> select_shape(dimensions);
    std::vector<int> input_shape(dimensions);
    std::vector<int> select_stride(dimensions);
    std::vector<int> input_stride(dimensions);

    for (int i = 0; i < dimensions; ++i) {
        if (i < select->dimensions()) {
            select_shape[i] = select->length(i);
        } else {
            select_shape[i] = 1;
        }
        input_shape[i] = inputs[1]->length(i);
    }

    for (int i = dimensions - 1; i >= 0; --i) {
        if (i == dimensions - 1) {
            select_stride[i] = 1;
            input_stride[i] = 1;
        } else {
            select_stride[i] = select_shape[i + 1] * select_stride[i + 1];
            input_stride[i] = input_shape[i + 1] * input_stride[i + 1];
        }
    }

    ApplySelectAtAxis(0, dimensions, selectPtr, input0Ptr, input1Ptr, outputPtr,
                      select_shape, input_shape, select_stride, input_stride);
    return NO_ERROR;
}

class CPUSelectCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUSelect(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUSelectCreator, OpType_Select);
} // namespace MNN
