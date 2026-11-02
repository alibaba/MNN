//
//  CoreMLScale.cpp
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLScale.hpp"

namespace MNN {

CoreMLScale::CoreMLScale(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CoreMLCommonExecution(b, op) {
}

ErrorCode CoreMLScale::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto scaleParam = mOp->main_as_Scale();
    auto mnnScale = scaleParam->scaleData();
    auto mnnBias = scaleParam->biasData();
    auto channel = scaleParam->channels();
    std::vector<int> broadcast(inputs[0]->dimensions(), 1);
    if (broadcast.empty() || channel <= 0 || !mnnScale || mnnScale->size() != channel ||
        (mnnBias && mnnBias->size() != 0 && mnnBias->size() != channel)) {
        return NOT_SUPPORT;
    }
    const int axis = TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NHWC
                         ? broadcast.size() - 1
                         : std::min<int>(1, broadcast.size() - 1);
    broadcast[axis] = channel;
    auto scale = buildBlob(mnnScale->data(), mnnScale->size() * sizeof(float),
                           CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32, broadcast, "scale");
    auto builder = operation("mul");
    builder.input("x", inputs[0]).input("y", scale);
    if (mnnBias && mnnBias->size() > 0) {
        auto mul_output = builder.output(inputs[0]->shape(), "mul");
        auto bias = buildBlob(mnnBias->data(), mnnBias->size() * sizeof(float),
                              CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32, broadcast, "bias");
        auto add_builder = operation("add");
        add_builder.input("x", mul_output).input("y", bias);
        add_builder.output(outputs);
    } else {
        builder.output(outputs);
    }
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLScale, OpType_Scale);
} // namespace MNN
