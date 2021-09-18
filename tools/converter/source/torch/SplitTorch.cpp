//
//  SplitTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(SplitTorch);

MNN::OpType SplitTorch::opType() {
    return MNN::OpType_Slice;
}
MNN::OpParameter SplitTorch::type() {
    return MNN::OpParameter_Slice;
}
std::vector<int> SplitTorch::inputTensorIdx() {
    return {0};
}

void SplitTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::SliceT;
    param->slicePoints.push_back(getValue<int64_t>(node->input(1)));
    param->axis = getValue<int64_t>(node->input(2));
    param->sourceType = MNN::NetSource_TENSORFLOW;
    dstOp->main.value  = param;
}

REGISTER_CONVERTER(SplitTorch, split);
