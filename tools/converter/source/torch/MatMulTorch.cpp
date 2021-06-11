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

void MatMulTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, torchContext* context) {
    auto param = new MNN::MatMulT;
    dstOp->main.value = param;
}

REGISTER_CONVERTER(MatMulTorch, matmul);

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

void AddmmTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, torchContext* context) {
    auto extra        = new MNN::ExtraT;
    dstOp->main.value = extra;
    extra->engine     = "Torch";
    extra->type       = node->kind().toUnqualString();
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
