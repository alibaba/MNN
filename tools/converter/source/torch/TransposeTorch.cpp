//
//  TransposeTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(TransposeTorch);

MNN::OpType TransposeTorch::opType() {
    return MNN::OpType_Permute;
}
MNN::OpParameter TransposeTorch::type() {
    return MNN::OpParameter_Permute;
}
std::vector<int> TransposeTorch::inputTensorIdx() {
    return {0};
}

void TransposeTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, torchContext* context) {
    auto param = new MNN::PermuteT;
    std::string opType = node->kind().toUnqualString();
    if (opType == "t") {
        param->dims = {1, 0};
    } else {
        // TODO: now just support dim = 5
        const auto inputs = node->inputs();
        int dim1 = getValue<int64_t>(inputs[1]);
        int dim2 = getValue<int64_t>(inputs[2]);
        param->dims = { 0, 1, 2, 3, 4 };
        param->dims[dim1] = dim2;
        param->dims[dim2] = dim1;
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(TransposeTorch, t);
REGISTER_CONVERTER(TransposeTorch, transpose);
