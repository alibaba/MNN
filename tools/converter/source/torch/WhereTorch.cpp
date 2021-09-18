//
//  WhereTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(WhereTorch);

MNN::OpType WhereTorch::opType() {
    return MNN::OpType_Select;
}
MNN::OpParameter WhereTorch::type() {
    return MNN::OpParameter_NONE;
}
std::vector<int> WhereTorch::inputTensorIdx() {
    return {-1};
}

void WhereTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    return;
}

REGISTER_CONVERTER(WhereTorch, where);
