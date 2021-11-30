//
//  ReluTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(ReluTorch);

MNN::OpType ReluTorch::opType() {
    return MNN::OpType_ReLU;
}
MNN::OpParameter ReluTorch::type() {
    return MNN::OpParameter_Relu;
}
std::vector<int> ReluTorch::inputTensorIdx() {
    return {0};
}

void ReluTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::ReluT;
    dstOp->main.value = param;
}

REGISTER_CONVERTER(ReluTorch, relu);

DECLARE_OP_CONVERTER(Relu6Torch);

MNN::OpType Relu6Torch::opType() {
    return MNN::OpType_ReLU6;
}
MNN::OpParameter Relu6Torch::type() {
    return MNN::OpParameter_Relu6;
}
std::vector<int> Relu6Torch::inputTensorIdx() {
    return {0};
}

void Relu6Torch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::Relu6T;
    if (getRealOpType(node) == "clamp") {
        param->minValue = getValue<double>(node->input(1));
        param->maxValue = getValue<double>(node->input(2));
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(Relu6Torch, hardtanh);
REGISTER_CONVERTER(Relu6Torch, clamp);
