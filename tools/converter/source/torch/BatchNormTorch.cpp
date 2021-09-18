//
//  BatchNormTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(BatchNormTorch);

MNN::OpType BatchNormTorch::opType() {
    return MNN::OpType_BatchNorm;
}
MNN::OpParameter BatchNormTorch::type() {
    return MNN::OpParameter_BatchNorm;
}
std::vector<int> BatchNormTorch::inputTensorIdx() {
    return {0};
}

void BatchNormTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::BatchNormT;
    const auto& inputs = node->inputs();
    const auto slope = inputs[1];
    const auto bias = inputs[2];
    const auto mean = inputs[3];
    const auto var = inputs[4];
    const auto epsilon = inputs[7];
    std::vector<int> shape;
    param->slopeData = getValue<float>(slope, shape);
    param->channels = shape[0];
    param->biasData = getValue<float>(bias, shape);
    param->meanData = getValue<float>(mean, shape);
    param->varData = getValue<float>(var, shape);
    param->epsilon = getValue<float>(epsilon);
    param->Adata = std::vector<float>(param->channels, 0.f);
    param->Bdata = std::vector<float>(param->channels, 0.f);
    dstOp->main.value = param;
}

REGISTER_CONVERTER(BatchNormTorch, batch_norm);
