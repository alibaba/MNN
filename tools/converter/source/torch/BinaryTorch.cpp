//
//  BinaryTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(BinaryTorch);

MNN::OpType BinaryTorch::opType() {
    return MNN::OpType_BinaryOp;
}

MNN::OpParameter BinaryTorch::type() {
    return MNN::OpParameter_BinaryOp;
}

std::vector<int> BinaryTorch::inputTensorIdx() {
    return {0, 1};
}

void BinaryTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, torchContext* context) {
    static std::map<std::string, MNN::BinaryOpOperation> gMaps{
        {"add", MNN::BinaryOpOperation_ADD}, {"add_", MNN::BinaryOpOperation_ADD},
        {"sum_", MNN::BinaryOpOperation_ADD},
        {"sub_", MNN::BinaryOpOperation_SUB}, {"div_", MNN::BinaryOpOperation_REALDIV},
        {"mul_", MNN::BinaryOpOperation_MUL}, {"pow_", MNN::BinaryOpOperation_POW},
        {"eq", MNN::BinaryOpOperation_EQUAL}, {"less_", MNN::BinaryOpOperation_LESS},
        {"greater_", MNN::BinaryOpOperation_GREATER}, {"max_", MNN::BinaryOpOperation_MAXIMUM},
        {"min_", MNN::BinaryOpOperation_MINIMUM}, {"floordiv", MNN::BinaryOpOperation_FLOORDIV},
    };
    auto param = new MNN::BinaryOpT;
    std::string opType = node->kind().toUnqualString();
    param->opType = gMaps[opType];
    dstOp->main.value = param;
}

REGISTER_CONVERTER(BinaryTorch, add);
REGISTER_CONVERTER(BinaryTorch, add_);
REGISTER_CONVERTER(BinaryTorch, eq);
REGISTER_CONVERTER(BinaryTorch, floordiv);
