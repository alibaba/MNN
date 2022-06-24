//
//  ZerosLikeTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/10/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(ZerosLikeTorch);

MNN::OpType ZerosLikeTorch::opType() {
    return MNN::OpType_ZerosLike;
}
MNN::OpParameter ZerosLikeTorch::type() {
    return MNN::OpParameter_NONE;
}
std::vector<int> ZerosLikeTorch::inputTensorIdx() {
    return {0};
}

void ZerosLikeTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    return;
}

REGISTER_CONVERTER(ZerosLikeTorch, zeros_like);

DECLARE_OP_CONVERTER(FullLikeTorch);

MNN::OpType FullLikeTorch::opType() {
    return MNN::OpType_Extra;
}
MNN::OpParameter FullLikeTorch::type() {
    return MNN::OpParameter_Extra;
}
std::vector<int> FullLikeTorch::inputTensorIdx() {
    return {0, 1};
}

void FullLikeTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto extra        = new MNN::ExtraT;
    extra->engine     = "Torch";
    extra->type       = "full_like";
    dstOp->main.value = extra;
    return;
}

REGISTER_CONVERTER(FullLikeTorch, full_like);

DECLARE_OP_CONVERTER(OnesLikeTorch);

MNN::OpType OnesLikeTorch::opType() {
    return MNN::OpType_Extra;
}
MNN::OpParameter OnesLikeTorch::type() {
    return MNN::OpParameter_Extra;
}
std::vector<int> OnesLikeTorch::inputTensorIdx() {
    return {0};
}

void OnesLikeTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto extra        = new MNN::ExtraT;
    extra->engine     = "Torch";
    extra->type       = "ones_like";
    dstOp->main.value = extra;
    return;
}

REGISTER_CONVERTER(OnesLikeTorch, ones_like);

DECLARE_OP_CONVERTER(ZerosTorch);

MNN::OpType ZerosTorch::opType() {
    return MNN::OpType_Extra;
}
MNN::OpParameter ZerosTorch::type() {
    return MNN::OpParameter_Extra;
}
std::vector<int> ZerosTorch::inputTensorIdx() {
    return {0};
}

void ZerosTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto extra        = new MNN::ExtraT;
    extra->engine     = "Torch";
    extra->type       = getRealOpType(node);
    dstOp->main.value = extra;
    return;
}

REGISTER_CONVERTER(ZerosTorch, zeros);
REGISTER_CONVERTER(ZerosTorch, ones);
