//
//  FlipTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(FlipTorch);

MNN::OpType FlipTorch::opType() {
    return MNN::OpType_Extra;
}
MNN::OpParameter FlipTorch::type() {
    return MNN::OpParameter_Extra;
}
std::vector<int> FlipTorch::inputTensorIdx() {
    return {0};
}

void FlipTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto extra = new MNN::ExtraT;
    dstOp->main.value = extra;
    extra->engine     = "Torch";
    extra->type       = "flip";
    extra->attr.resize(1);
    extra->attr[0].reset(new MNN::AttributeT);
    extra->attr[0]->key = "dims";
    auto dims = getValue<std::vector<int64_t>>(node->input(1));
    MNN_ASSERT(dims.size() == 1);
    extra->attr[0]->i = dims[0];
}

REGISTER_CONVERTER(FlipTorch, flip);
