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
