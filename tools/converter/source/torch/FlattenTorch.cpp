//
//  FlattenTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(FlattenTorch);

MNN::OpType FlattenTorch::opType() {
    return MNN::OpType_Flatten;
}
MNN::OpParameter FlattenTorch::type() {
    return MNN::OpParameter_Flatten;
}
std::vector<int> FlattenTorch::inputTensorIdx() {
    return {0};
}

void FlattenTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::FlattenT;
    const auto& inputs = node->inputs();
    const auto start_dim = inputs[1];
    const auto end_dim = inputs[2];
    param->axis = getValue<int64_t>(start_dim);
    param->endAxis = getValue<int64_t>(end_dim);
    dstOp->main.value = param;
}

REGISTER_CONVERTER(FlattenTorch, flatten);
