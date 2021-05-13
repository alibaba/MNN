//
//  CPUScatterNd.cpp
//  MNN
//
//  Created by MNN on 2019/11/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUScatterNd.hpp"
#include "backend/cpu/CPUBackend.hpp"

namespace MNN {

template <typename T>
void ScatterNdImpl(const Tensor* indices, const Tensor* updates, const Tensor* shape, Tensor* output) {
    const auto indicesPtr      = indices->host<int32_t>();
    const auto updatesPtr      = updates->host<T>();
    auto outputPtr             = output->host<T>();
    const int indicesDimension = indices->dimensions();
    const int indicesLastDim   = indices->length(indicesDimension - 1);
    const int indexes          = indices->elementSize() / indicesLastDim;
    int accNumber              = 1;
    for (int i = indicesDimension - 1; i < updates->dimensions(); ++i) {
        accNumber *= updates->length(i);
    }

    const int outputElementSize = output->elementSize();
    int remainSize              = outputElementSize;
    std::vector<int> dimsToCount(indicesLastDim, 0);
    for (int i = 0; i < indicesLastDim; ++i) {
        dimsToCount[i] = remainSize / output->length(i);
        remainSize     = dimsToCount[i];
    }

    for (int i = 0; i < indexes; ++i) {
        int pos = 0;
        bool valid = true;
        for (int j = 0; j < indicesLastDim; ++j) {
            auto curIndex = indicesPtr[i * indicesLastDim + j];
            if (curIndex < 0 || curIndex >= output->length(j)) {
                valid = false;
                break;
            }
            pos += curIndex * dimsToCount[j];
        }
        if (valid) {
            for (int k = 0; k < accNumber; ++k) {
                outputPtr[pos + k] += updatesPtr[i * accNumber + k];
            }
        }
    }
}

ErrorCode CPUScatterNd::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto indices         = inputs[0];
    auto updates         = inputs[1];
    auto shape           = inputs[2];
    auto output          = outputs[0];
    const int outputSize = output->size();

    auto outputRawPtr = output->host<int8_t>();
    memset(outputRawPtr, 0, outputSize);

    auto updatesDataType = updates->getType();
    if (updatesDataType == halide_type_of<int32_t>()) {
        ScatterNdImpl<int32_t>(indices, updates, shape, output);
    } else if (updatesDataType == halide_type_of<float>()) {
        ScatterNdImpl<float>(indices, updates, shape, output);
    } else {
        MNN_ERROR("TODO, ScatterNd support data type: %d\n", updatesDataType.code);
        return NOT_SUPPORT;
    }

    return NO_ERROR;
}

class CPUScatterNdCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUScatterNd(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUScatterNdCreator, OpType_ScatterNd);

} // namespace MNN
