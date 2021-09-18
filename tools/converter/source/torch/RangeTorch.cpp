//
//  RangeTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(RangeTorch);

MNN::OpType RangeTorch::opType() {
    return MNN::OpType_Range;
}
MNN::OpParameter RangeTorch::type() {
    return MNN::OpParameter_NONE;
}
std::vector<int> RangeTorch::inputTensorIdx() {
    return {0,1,2};
}

void RangeTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    return;
}

REGISTER_CONVERTER(RangeTorch, arange);
REGISTER_CONVERTER(RangeTorch, range);
