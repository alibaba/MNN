//
//  ShapeTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

// size -> Shape
DECLARE_OP_CONVERTER(ShapeTorch);

MNN::OpType ShapeTorch::opType() {
    return MNN::OpType_Extra;
}
MNN::OpParameter ShapeTorch::type() {
    return MNN::OpParameter_Extra;
}
std::vector<int> ShapeTorch::inputTensorIdx() {
    return {-1};
}

void ShapeTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto extra = new MNN::ExtraT;
    dstOp->main.value = extra;
    extra->engine     = "Torch";
    extra->type       = "size";
}

REGISTER_CONVERTER(ShapeTorch, size);

// dim -> Rank
DECLARE_OP_CONVERTER(RankTorch);

MNN::OpType RankTorch::opType() {
    return MNN::OpType_Rank;
}
MNN::OpParameter RankTorch::type() {
    return MNN::OpParameter_NONE;
}
std::vector<int> RankTorch::inputTensorIdx() {
    return {0};
}

void RankTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(RankTorch, dim);

// len -> Size
DECLARE_OP_CONVERTER(SizeTorch);

MNN::OpType SizeTorch::opType() {
    return MNN::OpType_Size;
}
MNN::OpParameter SizeTorch::type() {
    return MNN::OpParameter_NONE;
}
std::vector<int> SizeTorch::inputTensorIdx() {
    return {0};
}

void SizeTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(SizeTorch, len);
REGISTER_CONVERTER(SizeTorch, numel);
