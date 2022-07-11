//
//  GatherTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(GatherTorch);

MNN::OpType GatherTorch::opType() {
    return MNN::OpType_Gather;
}
MNN::OpParameter GatherTorch::type() {
    return MNN::OpParameter_Gather;
}
std::vector<int> GatherTorch::inputTensorIdx() {
    return {0, 1};
}

void GatherTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::GatherT;
    std::string opType = getRealOpType(node);;
    dstOp->main.value = param;
}

REGISTER_CONVERTER(GatherTorch, __getitem__);
REGISTER_CONVERTER(GatherTorch, embedding);

DECLARE_OP_CONVERTER(SelectTorch);

MNN::OpType SelectTorch::opType() {
    return MNN::OpType_Extra;
}
MNN::OpParameter SelectTorch::type() {
    return MNN::OpParameter_Extra;
}
std::vector<int> SelectTorch::inputTensorIdx() {
    return {-1};
}

void SelectTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto extra = new MNN::ExtraT;
    dstOp->main.value = extra;
    extra->engine     = "Torch";
    extra->type       = getRealOpType(node);
}

REGISTER_CONVERTER(SelectTorch, select);
REGISTER_CONVERTER(SelectTorch, index_select);
