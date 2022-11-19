//
//  UniformTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2022/11/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(UniformTorch);

MNN::OpType UniformTorch::opType() {
    return MNN::OpType_RandomUniform;
}
MNN::OpParameter UniformTorch::type() {
    return MNN::OpParameter_RandomUniform;
}
std::vector<int> UniformTorch::inputTensorIdx() {
    return {0};
}

void UniformTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::RandomUniformT;
    param->low = getValue<double>(node->input(1));
    param->high = getValue<double>(node->input(2));
    dstOp->main.value = param;
}

REGISTER_CONVERTER(UniformTorch, uniform);
