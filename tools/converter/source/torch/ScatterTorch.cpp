//
//  ScatterTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/06/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(ScatterTorch);

MNN::OpType ScatterTorch::opType() {
    return MNN::OpType_ScatterElements;
}
MNN::OpParameter ScatterTorch::type() {
    return MNN::OpParameter_BinaryOp;
}
std::vector<int> ScatterTorch::inputTensorIdx() {
    return {0, 2, 3, 1};
}

void ScatterTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::BinaryOpT;
    if (getRealOpType(node) == "scatter_add") {
        param->opType = MNN::BinaryOpOperation_ADD;
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(ScatterTorch, scatter);
REGISTER_CONVERTER(ScatterTorch, scatter_add);
