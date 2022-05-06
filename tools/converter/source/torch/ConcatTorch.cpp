//
//  ConcatTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(ListTorch);

MNN::OpType ListTorch::opType() {
    return MNN::OpType_Pack;
}
MNN::OpParameter ListTorch::type() {
    return MNN::OpParameter_PackParam;
}
std::vector<int> ListTorch::inputTensorIdx() {
    return {-1};
}

void ListTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::PackParamT;
    param->axis = 0;
    if (getRealOpType(node) == "stack") {
        dstOp->inputIndexes.pop_back();
        auto axis = node->inputs().back();
        param->axis = getValue<int64_t>(axis);
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(ListTorch, stack);
REGISTER_CONVERTER(ListTorch, ListConstruct);

DECLARE_OP_CONVERTER(TupleTorch);

MNN::OpType TupleTorch::opType() {
    return MNN::OpType_Concat;
}
MNN::OpParameter TupleTorch::type() {
    return MNN::OpParameter_Axis;
}
std::vector<int> TupleTorch::inputTensorIdx() {
    return {-1};
}

void TupleTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::AxisT;
    param->axis = 0;
    dstOp->main.value = param;
}

REGISTER_CONVERTER(TupleTorch, TupleConstruct);

DECLARE_OP_CONVERTER(ConcatTorch);

MNN::OpType ConcatTorch::opType() {
    return MNN::OpType_Concat;
}
MNN::OpParameter ConcatTorch::type() {
    return MNN::OpParameter_Axis;
}
std::vector<int> ConcatTorch::inputTensorIdx() {
    return {};
}

void ConcatTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::AxisT;
    const auto inputs = node->inputs();
    auto tensorlist = inputs[0];
    for (const auto input : tensorlist->node()->inputs()) {
        dstOp->inputIndexes.push_back(scope->lookupTensor(input->debugName()));
    }
    param->axis = getValue<int64_t>(inputs[1]);
    dstOp->main.value = param;
}

REGISTER_CONVERTER(ConcatTorch, cat);
