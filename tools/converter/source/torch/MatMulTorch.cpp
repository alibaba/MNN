//
//  MatMulTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(MatMulTorch);

MNN::OpType MatMulTorch::opType() {
    return MNN::OpType_MatMul;
}
MNN::OpParameter MatMulTorch::type() {
    return MNN::OpParameter_MatMul;
}
std::vector<int> MatMulTorch::inputTensorIdx() {
    return {0, 1};
}

void MatMulTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::MatMulT;
    std::string opType = getRealOpType(node);
    if (opType == "linear") {
        std::vector<int> shape;
        param->bias = getValue<float>(node->input(2), shape);
        param->transposeB = true;
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(MatMulTorch, matmul);
REGISTER_CONVERTER(MatMulTorch, linear);

DECLARE_OP_CONVERTER(BatchMatMulTorch);

MNN::OpType BatchMatMulTorch::opType() {
    return MNN::OpType_BatchMatMul;
}
MNN::OpParameter BatchMatMulTorch::type() {
    return MNN::OpParameter_BatchMatMulParam;
}
std::vector<int> BatchMatMulTorch::inputTensorIdx() {
    return {0, 1};
}

void BatchMatMulTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::BatchMatMulParamT;
    dstOp->main.value = param;
}

REGISTER_CONVERTER(BatchMatMulTorch, bmm);

DECLARE_OP_CONVERTER(AddmmTorch);

MNN::OpType AddmmTorch::opType() {
    return MNN::OpType_Extra;
}
MNN::OpParameter AddmmTorch::type() {
    return MNN::OpParameter_Extra;
}
std::vector<int> AddmmTorch::inputTensorIdx() {
    return {0, 1, 2};
}

void AddmmTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto extra        = new MNN::ExtraT;
    dstOp->main.value = extra;
    extra->engine     = "Torch";
    extra->type       = getRealOpType(node);
    const auto inputs = node->inputs();
    const auto beta   = inputs[3];
    const auto alpha  = inputs[4];
    extra->attr.resize(2);
    extra->attr[0].reset(new MNN::AttributeT);
    extra->attr[0]->key = "beta";
    extra->attr[0]->i = getValue<int64_t>(beta);
    extra->attr[1].reset(new MNN::AttributeT);
    extra->attr[1]->key = "alpha";
    extra->attr[1]->i = getValue<int64_t>(alpha);
}

REGISTER_CONVERTER(AddmmTorch, addmm);

DECLARE_OP_CONVERTER(EinsumTorch);

MNN::OpType EinsumTorch::opType() {
    return MNN::OpType_Extra;
}
MNN::OpParameter EinsumTorch::type() {
    return MNN::OpParameter_Extra;
}
std::vector<int> EinsumTorch::inputTensorIdx() {
    return {1};
}

void EinsumTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto extra        = new MNN::ExtraT;
    dstOp->main.value = extra;
    extra->engine     = "Torch";
    extra->type       = getRealOpType(node);
    const auto inputs = node->inputs();
    const auto beta   = inputs[3];
    const auto alpha  = inputs[4];
    extra->attr.resize(2);
    extra->attr[0].reset(new MNN::AttributeT);
    extra->attr[0]->key = "beta";
    extra->attr[0]->i = getValue<int64_t>(beta);
    extra->attr[1].reset(new MNN::AttributeT);
    extra->attr[1]->key = "alpha";
    extra->attr[1]->i = getValue<int64_t>(alpha);
}

REGISTER_CONVERTER(EinsumTorch, einsum);
