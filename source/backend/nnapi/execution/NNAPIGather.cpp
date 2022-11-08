//
//  NNAPIGather.cpp
//  MNN
//
//  Created by MNN on 2022/10/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NNAPIGather.hpp"

namespace MNN {


NNAPIGather::NNAPIGather(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NNAPICommonExecution(b, op) {
}

ErrorCode NNAPIGather::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    int axis = 0;
    if (inputs.size() == 3) {
        auto axis_tensor = inputs[2];
        axis = axis_tensor->host<int32_t>()[0];
    }
    if (mOp->main_type() == OpParameter_Axis) {
        axis = mOp->main_as_Axis()->axis();
    }
    if (axis < 0) {
        axis = input->buffer().dimensions + axis;
    }
    // gather: [input, axis, indices]
    auto inputIdx   = mNNAPIBackend->getTensorIdx(inputs[0]);
    auto axisIdx    = buildScalar(formatAxis(axis, input));
    auto indicesIdx = mNNAPIBackend->getTensorIdx(inputs[1]);
    return buildOperation(ANEURALNETWORKS_GATHER, {inputIdx, axisIdx, indicesIdx}, getTensorIdxs(outputs));
}

REGISTER_NNAPI_OP_CREATOR(NNAPIGather, OpType_GatherV2)
} // namespace MNN
