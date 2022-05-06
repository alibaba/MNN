//
//  BroadcastTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2022/04/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(BroadcastTorch);

MNN::OpType BroadcastTorch::opType() {
    return MNN::OpType_BroadcastTo;
}
MNN::OpParameter BroadcastTorch::type() {
    return MNN::OpParameter_NONE;
}
std::vector<int> BroadcastTorch::inputTensorIdx() {
    return {0, 1};
}

void BroadcastTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(BroadcastTorch, expand);

DECLARE_OP_CONVERTER(BroadcastAsTorch);

MNN::OpType BroadcastAsTorch::opType() {
    return MNN::OpType_Extra;
}
MNN::OpParameter BroadcastAsTorch::type() {
    return MNN::OpParameter_Extra;
}
std::vector<int> BroadcastAsTorch::inputTensorIdx() {
    return {0, 1};
}

void BroadcastAsTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto extra = new MNN::ExtraT;
    dstOp->main.value = extra;
    extra->engine     = "Torch";
    extra->type       = "broadcast_as";
}

REGISTER_CONVERTER(BroadcastAsTorch, expand_as);
