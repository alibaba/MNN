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

void UnaryTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    static std::map<std::string, MNN::UnaryOpOperation> gMaps{
        {"abs", MNN::UnaryOpOperation_ABS},
        {"neg", MNN::UnaryOpOperation_NEG},
        {"floor", MNN::UnaryOpOperation_FLOOR},
        {"ceil", MNN::UnaryOpOperation_CEIL},
        {"square", MNN::UnaryOpOperation_SQUARE},
        {"sqrt", MNN::UnaryOpOperation_SQRT},
        {"rsqrt", MNN::UnaryOpOperation_RSQRT},
        {"exp", MNN::UnaryOpOperation_EXP},
        {"log", MNN::UnaryOpOperation_LOG},
        {"sin", MNN::UnaryOpOperation_SIN},
        {"cos", MNN::UnaryOpOperation_COS},
        {"tan", MNN::UnaryOpOperation_TAN},
        {"asin", MNN::UnaryOpOperation_ASIN},
        {"acos", MNN::UnaryOpOperation_ACOS},
        {"atan", MNN::UnaryOpOperation_ATAN},
        {"reciprocal", MNN::UnaryOpOperation_RECIPROCAL},
        {"log1p", MNN::UnaryOpOperation_LOG1P},
        {"bernoulli", MNN::UnaryOpOperation_BNLL},
        {"acosh", MNN::UnaryOpOperation_ACOSH},
        {"sinh", MNN::UnaryOpOperation_SINH},
        {"asinh", MNN::UnaryOpOperation_ASINH},
        {"atanh", MNN::UnaryOpOperation_ATANH},
        {"sign", MNN::UnaryOpOperation_SIGN},
        {"round", MNN::UnaryOpOperation_ROUND},
        {"cosh", MNN::UnaryOpOperation_COSH},
        {"erf", MNN::UnaryOpOperation_ERF},
        {"erfc", MNN::UnaryOpOperation_ERFC},
        {"erfinv", MNN::UnaryOpOperation_ERFINV},
        {"expm1", MNN::UnaryOpOperation_EXPM1},
        {"tanh", MNN::UnaryOpOperation_TANH},
        {"sigmoid", MNN::UnaryOpOperation_SIGMOID},
        {"hardswish", MNN::UnaryOpOperation_HARDSWISH},
        {"gelu", MNN::UnaryOpOperation_GELU_STANDARD},
    };
    auto param = new MNN::UnaryOpT;
    std::string opType =  getRealOpType(node);
    param->opType = gMaps[opType];
    dstOp->main.value = param;
}

REGISTER_CONVERTER(UnaryTorch, abs);
REGISTER_CONVERTER(UnaryTorch, neg);
REGISTER_CONVERTER(UnaryTorch, floor);
REGISTER_CONVERTER(UnaryTorch, ceil);
REGISTER_CONVERTER(UnaryTorch, square);
REGISTER_CONVERTER(UnaryTorch, sqrt);
REGISTER_CONVERTER(UnaryTorch, rsqrt);
REGISTER_CONVERTER(UnaryTorch, exp);
REGISTER_CONVERTER(UnaryTorch, log);
REGISTER_CONVERTER(UnaryTorch, sin);
REGISTER_CONVERTER(UnaryTorch, cos);
REGISTER_CONVERTER(UnaryTorch, tan);
REGISTER_CONVERTER(UnaryTorch, asin);
REGISTER_CONVERTER(UnaryTorch, acos);
REGISTER_CONVERTER(UnaryTorch, atan);
REGISTER_CONVERTER(UnaryTorch, reciprocal);
REGISTER_CONVERTER(UnaryTorch, log1p);
REGISTER_CONVERTER(UnaryTorch, bernoulli);
REGISTER_CONVERTER(UnaryTorch, acosh);
REGISTER_CONVERTER(UnaryTorch, sinh);
REGISTER_CONVERTER(UnaryTorch, asinh);
REGISTER_CONVERTER(UnaryTorch, atanh);
REGISTER_CONVERTER(UnaryTorch, sign);
REGISTER_CONVERTER(UnaryTorch, round);
REGISTER_CONVERTER(UnaryTorch, cosh);
REGISTER_CONVERTER(UnaryTorch, erf);
REGISTER_CONVERTER(UnaryTorch, erfc);
REGISTER_CONVERTER(UnaryTorch, erfinv);
REGISTER_CONVERTER(UnaryTorch, expm1);
REGISTER_CONVERTER(UnaryTorch, tanh);
REGISTER_CONVERTER(UnaryTorch, sigmoid);
REGISTER_CONVERTER(UnaryTorch, hardswish);
REGISTER_CONVERTER(UnaryTorch, gelu);

// TODO: silu will impl as unary ?
DECLARE_OP_CONVERTER(ExtraUnaryTorch);

MNN::OpType ExtraUnaryTorch::opType() {
    return MNN::OpType_Extra;
}
MNN::OpParameter ExtraUnaryTorch::type() {
    return MNN::OpParameter_Extra;
}
std::vector<int> ExtraUnaryTorch::inputTensorIdx() {
    return {0};
}

void ExtraUnaryTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto extra        = new MNN::ExtraT;
    extra->engine     = "Torch";
    auto type         = getRealOpType(node);
    extra->type       = type;
    dstOp->main.value = extra;
    if (type == "softplus") {
        extra->attr.resize(2);
        extra->attr[0].reset(new MNN::AttributeT);
        extra->attr[0]->key = "beta";
        extra->attr[0]->i = getValue<int64_t>(node->input(1));
        extra->attr[1].reset(new MNN::AttributeT);
        extra->attr[1]->key = "threshold";
        extra->attr[1]->i = getValue<int64_t>(node->input(2));
    }
}

REGISTER_CONVERTER(ExtraUnaryTorch, silu);
REGISTER_CONVERTER(ExtraUnaryTorch, softplus);
REGISTER_CONVERTER(ExtraUnaryTorch, bitwise_not);
