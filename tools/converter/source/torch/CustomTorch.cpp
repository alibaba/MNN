//
//  CustomTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(CustomTorch);

MNN::OpType CustomTorch::opType() {
    return MNN::OpType_Plugin;
}
MNN::OpParameter CustomTorch::type() {
    return MNN::OpParameter_Plugin;
}
std::vector<int> CustomTorch::inputTensorIdx() {
    return {-1};
}

void CustomTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::PluginT;
    param->type = node->kind().toUnqualString();
    dstOp->main.value = param;
}

REGISTER_CONVERTER(CustomTorch, __custom__);
