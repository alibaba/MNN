//
//  UnaryOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(UnaryOnnx);

MNN::OpType UnaryOnnx::opType() {
    return MNN::OpType_UnaryOp;
}

MNN::OpParameter UnaryOnnx::type() {
    return MNN::OpParameter_UnaryOp;
}

void UnaryOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                    std::vector<const onnx::TensorProto *> initializers) {
    std::unique_ptr<MNN::UnaryOpT> unaryOpParam(new MNN::UnaryOpT);
    unaryOpParam->T = MNN::DataType_DT_FLOAT;

    const auto &originalType = onnxNode->op_type();

#define TO_UNARY_OP(src, dst)       \
    if (originalType == src) {      \
        unaryOpParam->opType = dst; \
    }

    TO_UNARY_OP("Floor", MNN::UnaryOpOperation_FLOOR);
    TO_UNARY_OP("Neg", MNN::UnaryOpOperation_NEG);
    TO_UNARY_OP("Abs", MNN::UnaryOpOperation_ABS);
    TO_UNARY_OP("Exp", MNN::UnaryOpOperation_EXP);
    TO_UNARY_OP("Cos", MNN::UnaryOpOperation_COS);
    TO_UNARY_OP("Sin", MNN::UnaryOpOperation_SIN);
    TO_UNARY_OP("Sqrt", MNN::UnaryOpOperation_SQRT);
    TO_UNARY_OP("Ceil", MNN::UnaryOpOperation_CEIL);
    TO_UNARY_OP("Log", MNN::UnaryOpOperation_LOG);
    TO_UNARY_OP("Tan", MNN::UnaryOpOperation_TAN);
    TO_UNARY_OP("ATan", MNN::UnaryOpOperation_ATAN);
    TO_UNARY_OP("Asin", MNN::UnaryOpOperation_ASIN);

    dstOp->main.value = unaryOpParam.release();
}

REGISTER_CONVERTER(UnaryOnnx, Floor);
REGISTER_CONVERTER(UnaryOnnx, Abs);
REGISTER_CONVERTER(UnaryOnnx, Neg);
REGISTER_CONVERTER(UnaryOnnx, Exp);
REGISTER_CONVERTER(UnaryOnnx, Cos);
REGISTER_CONVERTER(UnaryOnnx, Sin);
REGISTER_CONVERTER(UnaryOnnx, Sqrt);
REGISTER_CONVERTER(UnaryOnnx, Ceil);
REGISTER_CONVERTER(UnaryOnnx, Log);
REGISTER_CONVERTER(UnaryOnnx, Tan);
REGISTER_CONVERTER(UnaryOnnx, ATan);
REGISTER_CONVERTER(UnaryOnnx, Asin);
