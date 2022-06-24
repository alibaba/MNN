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

void BinaryTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    static std::map<std::string, MNN::BinaryOpOperation> gMaps{
        {"add", MNN::BinaryOpOperation_ADD}, {"sum", MNN::BinaryOpOperation_ADD},
        {"sub", MNN::BinaryOpOperation_SUB}, {"rsub", MNN::BinaryOpOperation_SUB},
        {"mul", MNN::BinaryOpOperation_MUL},
        {"pow", MNN::BinaryOpOperation_POW},
        {"div", MNN::BinaryOpOperation_REALDIV},
        {"min_compare", MNN::BinaryOpOperation_MINIMUM}, {"minimum", MNN::BinaryOpOperation_MINIMUM},
        {"max_compare", MNN::BinaryOpOperation_MAXIMUM}, {"maximum", MNN::BinaryOpOperation_MAXIMUM},
        {"gt", MNN::BinaryOpOperation_GREATER}, {"greater", MNN::BinaryOpOperation_GREATER},
        {"ge", MNN::BinaryOpOperation_GREATER_EQUAL},
        {"lt", MNN::BinaryOpOperation_LESS}, {"less", MNN::BinaryOpOperation_LESS},
        {"floordiv", MNN::BinaryOpOperation_FLOORDIV}, {"floor_divide", MNN::BinaryOpOperation_FLOORDIV},
        {"le", MNN::BinaryOpOperation_LESS_EQUAL},
        {"eq", MNN::BinaryOpOperation_EQUAL}, {"__is__", MNN::BinaryOpOperation_EQUAL},
        {"mode", MNN::BinaryOpOperation_MOD}, {"remainder", MNN::BinaryOpOperation_MOD},
        {"atan2", MNN::BinaryOpOperation_ATAN2},
        {"logical_or", MNN::BinaryOpOperation_LOGICALOR},
        {"__or__", MNN::BinaryOpOperation_BITWISE_OR}, {"__ior__", MNN::BinaryOpOperation_BITWISE_OR},
        {"__and__", MNN::BinaryOpOperation_BITWISE_AND}, {"__iand__", MNN::BinaryOpOperation_BITWISE_AND},
        {"__xor__", MNN::BinaryOpOperation_BITWISE_XOR}, {"__ixor__", MNN::BinaryOpOperation_BITWISE_XOR},
        {"ne", MNN::BinaryOpOperation_NOTEQUAL}, {"__isnot__", MNN::BinaryOpOperation_NOTEQUAL}
    };
    auto param = new MNN::BinaryOpT;
    std::string opType = getRealOpType(node);
    param->opType = gMaps[opType];
    dstOp->main.value = param;
    if (opType == "rsub") {
        MNN_ASSERT(getValue<int64_t>(node->input(2)) == 1);
        int x = dstOp->inputIndexes[0];
        dstOp->inputIndexes[0] = dstOp->inputIndexes[1];
        dstOp->inputIndexes[1] = x;
    }
}

REGISTER_CONVERTER(BinaryTorch, add);
REGISTER_CONVERTER(BinaryTorch, sum_binary);
REGISTER_CONVERTER(BinaryTorch, sub);
REGISTER_CONVERTER(BinaryTorch, mul);
REGISTER_CONVERTER(BinaryTorch, pow);
REGISTER_CONVERTER(BinaryTorch, div);
REGISTER_CONVERTER(BinaryTorch, min_binary);
REGISTER_CONVERTER(BinaryTorch, minimum);
REGISTER_CONVERTER(BinaryTorch, max_binary);
REGISTER_CONVERTER(BinaryTorch, maximum);
REGISTER_CONVERTER(BinaryTorch, gt);
REGISTER_CONVERTER(BinaryTorch, greater);
REGISTER_CONVERTER(BinaryTorch, ge);
REGISTER_CONVERTER(BinaryTorch, lt);
REGISTER_CONVERTER(BinaryTorch, less);
REGISTER_CONVERTER(BinaryTorch, floordiv);
REGISTER_CONVERTER(BinaryTorch, floor_divide);
REGISTER_CONVERTER(BinaryTorch, le);
REGISTER_CONVERTER(BinaryTorch, eq);
REGISTER_CONVERTER(BinaryTorch, mode);
REGISTER_CONVERTER(BinaryTorch, remainder);
REGISTER_CONVERTER(BinaryTorch, atan2);
REGISTER_CONVERTER(BinaryTorch, logical_or);
REGISTER_CONVERTER(BinaryTorch, ne);
REGISTER_CONVERTER(BinaryTorch, rsub);
REGISTER_CONVERTER(BinaryTorch, __is__);
REGISTER_CONVERTER(BinaryTorch, __isnot__);
REGISTER_CONVERTER(BinaryTorch, __or__);
REGISTER_CONVERTER(BinaryTorch, __ior__);
REGISTER_CONVERTER(BinaryTorch, __and__);
REGISTER_CONVERTER(BinaryTorch, __iand__);
REGISTER_CONVERTER(BinaryTorch, __xor__);
REGISTER_CONVERTER(BinaryTorch, __ixor__);
