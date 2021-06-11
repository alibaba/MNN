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

void GatherTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, torchContext* context) {
    auto param = new MNN::GatherT;
    dstOp->main.value = param;
}

REGISTER_CONVERTER(GatherTorch, __getitem__);
