//
//  EluTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2024/11/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(EluTorch);

MNN::OpType EluTorch::opType() {
    return MNN::OpType_ELU;
}
MNN::OpParameter EluTorch::type() {
    return MNN::OpParameter_ELU;
}
std::vector<int> EluTorch::inputTensorIdx() {
    return {0};
}

void EluTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::ELUT;
    if (node->inputs().size() > 1) {
        param->alpha = getValue<double>(node->input(1));
    } else {
        param->alpha = 1.0f;
    }
    dstOp->main.value = param;
    return;
}

REGISTER_CONVERTER(EluTorch, elu);
