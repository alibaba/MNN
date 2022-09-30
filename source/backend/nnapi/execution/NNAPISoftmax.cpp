//
//  NNAPISoftmax.cpp
//  MNN
//
//  Created by MNN on 2022/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NNAPISoftmax.hpp"

namespace MNN {


NNAPISoftmax::NNAPISoftmax(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NNAPICommonExecution(b, op) {
}

ErrorCode NNAPISoftmax::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    int axis = mOp->main_as_Axis()->axis();
    auto inputIdxs = getTensorIdxs(inputs);
    // NNAPI Softmax inputs: [input, beta, axis]
    float beta = 1.0;
    inputIdxs.push_back(buildScalar(beta));
    bool needAxis = false;
    auto dims = inputs[0]->shape();
    for (int i = 0; i < dims.size(); i++) {
        if (i != axis && dims[i] > 1) {
            needAxis = true;
            break;
        }
    }
    if (needAxis) {
        inputIdxs.push_back(buildScalar(axis));
    }
    return buildOperation(ANEURALNETWORKS_SOFTMAX, inputIdxs, getTensorIdxs(outputs));
}

REGISTER_NNAPI_OP_CREATOR(NNAPISoftmax, OpType_Softmax)
} // namespace MNN
