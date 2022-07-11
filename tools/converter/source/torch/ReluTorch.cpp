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
    if (getRealOpType(node) == "leaky_relu") {
        if (node->inputs().size() > 1) {
            param->slope = getValue<double>(node->input(1));
        } else {
            param->slope = 0.01;
        }
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(ReluTorch, relu);
REGISTER_CONVERTER(ReluTorch, leaky_relu);

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
    bool isFloat = node->input(1)->type()->kind() == c10::TypeKind::FloatType;
    if (getRealOpType(node) == "clamp" || getRealOpType(node) == "hardtanh") {
        if (isFloat) {
            param->minValue = getValue<double>(node->input(1));
            param->maxValue = getValue<double>(node->input(2));
        } else {
            param->minValue = getValue<int64_t>(node->input(1));
            param->maxValue = getValue<int64_t>(node->input(2));
        }
    } else if (getRealOpType(node) == "clamp_min") {
        if (isFloat) {
            param->minValue = getValue<double>(node->input(1));
        } else {
            param->minValue = getValue<int64_t>(node->input(1));
        }
        param->maxValue = std::numeric_limits<float>::max();
    } else if (getRealOpType(node) == "clamp_max") {
        param->minValue = std::numeric_limits<float>::min();
        if (isFloat) {
            param->maxValue = getValue<double>(node->input(1));
        } else {
            param->maxValue = getValue<int64_t>(node->input(1));
        }
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(Relu6Torch, hardtanh);
REGISTER_CONVERTER(Relu6Torch, clamp);
REGISTER_CONVERTER(Relu6Torch, clamp_min);
REGISTER_CONVERTER(Relu6Torch, clamp_max);
