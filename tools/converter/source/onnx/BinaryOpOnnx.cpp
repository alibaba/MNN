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
    auto param = new MNN::BinaryOpT;
    static std::map<std::string, MNN::BinaryOpOperation> gMaps {
        {"Add", MNN::BinaryOpOperation_ADD},
        {"Sum", MNN::BinaryOpOperation_ADD},
        {"Sub", MNN::BinaryOpOperation_SUB},
        {"Div", MNN::BinaryOpOperation_REALDIV},
        {"Mul", MNN::BinaryOpOperation_MUL},
        {"Pow", MNN::BinaryOpOperation_POW},
    };

    auto type = onnxNode->op_type();
    param->opType = gMaps[type];
    param->T          = MNN::DataType_DT_FLOAT;
    dstOp->main.value = param;
}

REGISTER_CONVERTER(BinaryOpOnnx, Sum);
REGISTER_CONVERTER(BinaryOpOnnx, Add);
REGISTER_CONVERTER(BinaryOpOnnx, Sub);
REGISTER_CONVERTER(BinaryOpOnnx, Div);
REGISTER_CONVERTER(BinaryOpOnnx, Mul);
REGISTER_CONVERTER(BinaryOpOnnx, Pow);
