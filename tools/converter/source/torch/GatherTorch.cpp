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
    return MNN::OpType_Gather;
}
MNN::OpParameter SelectTorch::type() {
    return MNN::OpParameter_Axis;
}
std::vector<int> SelectTorch::inputTensorIdx() {
    return {0, 2};
}

void SelectTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::AxisT;
    param->axis = getValue<int64_t>(node->input(1));
    dstOp->main.value = param;
}

REGISTER_CONVERTER(SelectTorch, select);
