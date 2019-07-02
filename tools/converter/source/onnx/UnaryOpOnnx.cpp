//
//  UnaryOpOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(UnaryOpOnnx);

MNN::OpType UnaryOpOnnx::opType() {
    return MNN::OpType_UnaryOp;
}

MNN::OpParameter UnaryOpOnnx::type() {
    return MNN::OpParameter_UnaryOp;
}

void UnaryOpOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                       std::vector<const onnx::TensorProto*> initializers) {
    auto param = new MNN::UnaryOpT;
    static std::map<std::string, MNN::UnaryOpOperation> gMaps {
        {"Neg", MNN::UnaryOpOperation_NEG},
        {"Abs", MNN::UnaryOpOperation_ABS},
        {"Exp", MNN::UnaryOpOperation_EXP},
        {"Cos", MNN::UnaryOpOperation_COS},
        {"Sin", MNN::UnaryOpOperation_SIN},
        {"Sqrt", MNN::UnaryOpOperation_SQRT},
    };

    auto type = onnxNode->op_type();
    param->opType = gMaps[type];
    param->T          = MNN::DataType_DT_FLOAT;
    dstOp->main.value = param;
}

REGISTER_CONVERTER(UnaryOpOnnx, Abs);
REGISTER_CONVERTER(UnaryOpOnnx, Neg);
REGISTER_CONVERTER(UnaryOpOnnx, Exp);
REGISTER_CONVERTER(UnaryOpOnnx, Cos);
REGISTER_CONVERTER(UnaryOpOnnx, Sin);
REGISTER_CONVERTER(UnaryOpOnnx, Sqrt);
