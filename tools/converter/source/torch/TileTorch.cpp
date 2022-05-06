//
//  TileTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/07/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(TileTorch);

MNN::OpType TileTorch::opType() {
    return MNN::OpType_Tile;
}
MNN::OpParameter TileTorch::type() {
    return MNN::OpParameter_NONE;
}
std::vector<int> TileTorch::inputTensorIdx() {
    return {0, 1};
}

void TileTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    return;
}

REGISTER_CONVERTER(TileTorch, repeat);
