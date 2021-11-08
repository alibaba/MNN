//
//  PadTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(PadTorch);

MNN::OpType PadTorch::opType() {
    return MNN::OpType_Extra;
}
MNN::OpParameter PadTorch::type() {
    return MNN::OpParameter_Extra;
}
std::vector<int> PadTorch::inputTensorIdx() {
    return {-1};
}

void PadTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto extra        = new MNN::ExtraT;
    dstOp->main.value = extra;
    extra->engine     = "Torch";
    extra->type       = "pad";
    extra->attr.resize(1);
    extra->attr[0].reset(new MNN::AttributeT);
    extra->attr[0]->key = "mode";
    std::string opType = getRealOpType(node);
    if (opType.find("constant") != std::string::npos) {
        extra->attr[0]->s = "constant";
    } else if (opType.find("reflection") != std::string::npos) {
        extra->attr[0]->s = "reflect";
    }
}

REGISTER_CONVERTER(PadTorch, constant_pad_nd);
REGISTER_CONVERTER(PadTorch, reflection_pad1d);
REGISTER_CONVERTER(PadTorch, reflection_pad2d);
