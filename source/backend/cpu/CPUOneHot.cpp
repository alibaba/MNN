//
//  CPUOneHot.cpp
//  MNN
//
//  Created by MNN on 2019/11/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUOneHot.hpp"
#include "backend/cpu/CPUBackend.hpp"

namespace MNN {

template <typename T>
void OneHotImpl(int depth, int outerSize, int innerSize, const int* indices, const Tensor* onValueTensor,
                const Tensor* offValueTensor, Tensor* outputTensor) {
    const T onValue  = onValueTensor->host<T>()[0];
    const T offValue = offValueTensor->host<T>()[0];
    T* outputPtr     = outputTensor->host<T>();

    for (int i = 0; i < outerSize; ++i) {
        for (int j = 0; j < depth; ++j) {
            for (int k = 0; k < innerSize; ++k) {
                auto index = indices[i * innerSize + k];
                if (index == j) {
                    *outputPtr = onValue;
                } else {
                    *outputPtr = offValue;
                }
                outputPtr++;
            }
        }
    }
}

ErrorCode CPUOneHot::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto indices        = inputs[0];
    auto depthTensor    = inputs[1];
    auto onValueTensor  = inputs[2];
    auto offValueTensor = inputs[3];

    int axis = mAxis;
    if (axis < 0) {
        axis += outputs[0]->dimensions();
    }
    int outerSize = 1;
    for (int i = 0; i < axis; ++i) {
        outerSize *= indices->length(i);
    }
    const int depth       = depthTensor->host<int>()[0];
    const int innerSize   = indices->elementSize() / outerSize;
    const auto indicesPtr = indices->host<int>();

    auto dataType    = onValueTensor->getType();
    MNN_ASSERT(offValueTensor->getType() == dataType);

    if (dataType == halide_type_of<float>()) {
        OneHotImpl<float>(depth, outerSize, innerSize, indicesPtr, onValueTensor, offValueTensor, outputs[0]);
    } else if (dataType == halide_type_of<int>()) {
        OneHotImpl<int>(depth, outerSize, innerSize, indicesPtr, onValueTensor, offValueTensor, outputs[0]);
    } else {
        return NOT_SUPPORT;
    }
    return NO_ERROR;
}

class CPUOneHotCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUOneHot(backend, op->main_as_OneHotParam()->axis());
    }
};

REGISTER_CPU_OP_CREATOR(CPUOneHotCreator, OpType_OneHot);

} // namespace MNN
