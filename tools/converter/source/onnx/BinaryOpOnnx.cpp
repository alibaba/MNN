//
//  BinaryOpOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(BinaryOpOnnx);

MNN::OpType BinaryOpOnnx::opType() {
    return MNN::OpType_BinaryOp;
}

MNN::OpParameter BinaryOpOnnx::type() {
    return MNN::OpParameter_BinaryOp;
}

void BinaryOpOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                       std::vector<const onnx::TensorProto*> initializers) {
    const auto &originalType = onnxNode->op_type();
    auto param = new MNN::BinaryOpT;
#define TO_BINARY_OP(src, dst)       \
    if (originalType == src) {      \
        param->opType = dst; \
    }
    
    TO_BINARY_OP("Add", MNN::BinaryOpOperation_ADD);
    TO_BINARY_OP("And", MNN::BinaryOpOperation_MUL);
    TO_BINARY_OP("Div", MNN::BinaryOpOperation_REALDIV);
    TO_BINARY_OP("Mul", MNN::BinaryOpOperation_MUL);
    TO_BINARY_OP("Equal", MNN::BinaryOpOperation_EQUAL);
    TO_BINARY_OP("Less", MNN::BinaryOpOperation_LESS);
    TO_BINARY_OP("LessOrEqual", MNN::BinaryOpOperation_LESS_EQUAL);
    TO_BINARY_OP("Greater", MNN::BinaryOpOperation_GREATER);
    TO_BINARY_OP("GreaterOrEqual", MNN::BinaryOpOperation_GREATER_EQUAL);
    TO_BINARY_OP("Max", MNN::BinaryOpOperation_MAXIMUM);
    TO_BINARY_OP("Min", MNN::BinaryOpOperation_MINIMUM);
    // TODO: Consified fmod case
    TO_BINARY_OP("Mod", MNN::BinaryOpOperation_MOD);
    TO_BINARY_OP("Pow", MNN::BinaryOpOperation_POW);
    TO_BINARY_OP("Sub", MNN::BinaryOpOperation_SUB);
    TO_BINARY_OP("Sum", MNN::BinaryOpOperation_ADD);
    auto type         = onnxNode->op_type();
    dstOp->main.value = param;
}

REGISTER_CONVERTER(BinaryOpOnnx, Add);
REGISTER_CONVERTER(BinaryOpOnnx, And);
REGISTER_CONVERTER(BinaryOpOnnx, Sum);
REGISTER_CONVERTER(BinaryOpOnnx, Sub);
REGISTER_CONVERTER(BinaryOpOnnx, Div);
REGISTER_CONVERTER(BinaryOpOnnx, Mul);
REGISTER_CONVERTER(BinaryOpOnnx, Pow);
REGISTER_CONVERTER(BinaryOpOnnx, Equal);
REGISTER_CONVERTER(BinaryOpOnnx, Less);
REGISTER_CONVERTER(BinaryOpOnnx, LessOrEqual);
REGISTER_CONVERTER(BinaryOpOnnx, Greater);
REGISTER_CONVERTER(BinaryOpOnnx, GreaterOrEqual);
REGISTER_CONVERTER(BinaryOpOnnx, Max);
REGISTER_CONVERTER(BinaryOpOnnx, Min);
REGISTER_CONVERTER(BinaryOpOnnx, Mod);
