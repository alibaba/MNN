//
//  UnaryTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(UnaryTorch);

MNN::OpType UnaryTorch::opType() {
    return MNN::OpType_UnaryOp;
}

MNN::OpParameter UnaryTorch::type() {
    return MNN::OpParameter_UnaryOp;
}

std::vector<int> UnaryTorch::inputTensorIdx() {
    return {0};
}

void UnaryTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, torchContext* context) {
    static std::map<std::string, MNN::UnaryOpOperation> gMaps{
        {"abs", MNN::UnaryOpOperation_ABS}, {"ne", MNN::UnaryOpOperation_NEG},
        {"hardtanh_", MNN::UnaryOpOperation_TANH},
    };
    auto param = new MNN::UnaryOpT;
    std::string opType = node->kind().toUnqualString();
    param->opType = gMaps[opType];
    dstOp->main.value = param;
}

REGISTER_CONVERTER(UnaryTorch, ne);
