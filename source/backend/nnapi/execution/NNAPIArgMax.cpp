//
//  NNAPIArgMax.cpp
//  MNN
//
//  Created by MNN on 2022/10/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NNAPIArgMax.hpp"

namespace MNN {


NNAPIArgMax::NNAPIArgMax(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NNAPICommonExecution(b, op) {
}

ErrorCode NNAPIArgMax::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto output = outputs[0];
    auto param = mOp->main_as_ArgMax();
    int axis = param->axis();
    int type = ANEURALNETWORKS_ARGMAX;
    if (mOp->type() == OpType_ArgMin) {
        type = ANEURALNETWORKS_ARGMIN;
    }
    // argmax/argmin: [input, axis]
    auto inputIdxs = getTensorIdxs(inputs);
    inputIdxs.push_back(buildScalar(axis));
    return buildOperation(type, inputIdxs, getTensorIdxs(outputs));
}

REGISTER_NNAPI_OP_CREATOR(NNAPIArgMax, OpType_ArgMax)
REGISTER_NNAPI_OP_CREATOR(NNAPIArgMax, OpType_ArgMin)
} // namespace MNN
