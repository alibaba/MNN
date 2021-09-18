//
//  SqueezeTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(UnSqueezeTorch);

MNN::OpType UnSqueezeTorch::opType() {
    return MNN::OpType_Unsqueeze;
}
MNN::OpParameter UnSqueezeTorch::type() {
    return MNN::OpParameter_SqueezeParam;
}
std::vector<int> UnSqueezeTorch::inputTensorIdx() {
    return {0};
}

void UnSqueezeTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::SqueezeParamT;
    param->squeezeDims.push_back(getValue<int64_t>(node->input(1)));
    dstOp->main.value = param;
}

REGISTER_CONVERTER(UnSqueezeTorch, unsqueeze);
