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
                    OnnxScope* scope) {
    std::unique_ptr<MNN::UnaryOpT> unaryOpParam(new MNN::UnaryOpT);
    unaryOpParam->T = MNN::DataType_DT_FLOAT;

    const auto &originalType = onnxNode->op_type();

#define TO_UNARY_OP(src, dst)       \
    if (originalType == src) {      \
        unaryOpParam->opType = dst; \
    }

    TO_UNARY_OP("Abs", MNN::UnaryOpOperation_ABS);
    TO_UNARY_OP("Acos", MNN::UnaryOpOperation_ACOS);
    TO_UNARY_OP("Acosh", MNN::UnaryOpOperation_ACOSH);
    TO_UNARY_OP("Asinh", MNN::UnaryOpOperation_ASINH);
    TO_UNARY_OP("Atan", MNN::UnaryOpOperation_ATAN);
    TO_UNARY_OP("Atanh", MNN::UnaryOpOperation_ATANH);
    TO_UNARY_OP("Asin", MNN::UnaryOpOperation_ASIN);
    TO_UNARY_OP("Ceil", MNN::UnaryOpOperation_CEIL);
    TO_UNARY_OP("Cos", MNN::UnaryOpOperation_COS);
    TO_UNARY_OP("Cosh", MNN::UnaryOpOperation_COSH);
    TO_UNARY_OP("Exp", MNN::UnaryOpOperation_EXP);
    TO_UNARY_OP("Erf", MNN::UnaryOpOperation_ERF);
    TO_UNARY_OP("Erfc", MNN::UnaryOpOperation_ERFC);
    TO_UNARY_OP("Erfinv", MNN::UnaryOpOperation_ERFINV);
    TO_UNARY_OP("Expm1", MNN::UnaryOpOperation_EXPM1);
    TO_UNARY_OP("Floor", MNN::UnaryOpOperation_FLOOR);
    TO_UNARY_OP("HardSwish", MNN::UnaryOpOperation_HARDSWISH);
    TO_UNARY_OP("Log", MNN::UnaryOpOperation_LOG);
    TO_UNARY_OP("Log1p", MNN::UnaryOpOperation_LOG1P);
    TO_UNARY_OP("Gelu", MNN::UnaryOpOperation_GELU);
    TO_UNARY_OP("Neg", MNN::UnaryOpOperation_NEG);
    TO_UNARY_OP("Sin", MNN::UnaryOpOperation_SIN);
    TO_UNARY_OP("Sinh", MNN::UnaryOpOperation_SINH);
    TO_UNARY_OP("Sqrt", MNN::UnaryOpOperation_SQRT);
    TO_UNARY_OP("Tan", MNN::UnaryOpOperation_TAN);
    TO_UNARY_OP("Tanh", MNN::UnaryOpOperation_TANH);
    TO_UNARY_OP("Reciprocal", MNN::UnaryOpOperation_RECIPROCAL);
    TO_UNARY_OP("Round", MNN::UnaryOpOperation_ROUND);
    TO_UNARY_OP("Sign", MNN::UnaryOpOperation_SIGN);

    // For specitial error onnx
    TO_UNARY_OP("ATan", MNN::UnaryOpOperation_ATAN);
    dstOp->main.value = unaryOpParam.release();
}

REGISTER_CONVERTER(UnaryOnnx, Abs);
REGISTER_CONVERTER(UnaryOnnx, Acos);
REGISTER_CONVERTER(UnaryOnnx, Acosh);
REGISTER_CONVERTER(UnaryOnnx, Asinh);
REGISTER_CONVERTER(UnaryOnnx, Atan);
REGISTER_CONVERTER(UnaryOnnx, Atanh);
REGISTER_CONVERTER(UnaryOnnx, Asin);
REGISTER_CONVERTER(UnaryOnnx, Ceil);
REGISTER_CONVERTER(UnaryOnnx, Cos);
REGISTER_CONVERTER(UnaryOnnx, Cosh);
REGISTER_CONVERTER(UnaryOnnx, Expm1);
REGISTER_CONVERTER(UnaryOnnx, Exp);
REGISTER_CONVERTER(UnaryOnnx, Erf);
REGISTER_CONVERTER(UnaryOnnx, Erfc);
REGISTER_CONVERTER(UnaryOnnx, Erfinv);
REGISTER_CONVERTER(UnaryOnnx, Floor);
REGISTER_CONVERTER(UnaryOnnx, HardSwish);
REGISTER_CONVERTER(UnaryOnnx, Log);
REGISTER_CONVERTER(UnaryOnnx, Log1p);
REGISTER_CONVERTER(UnaryOnnx, Gelu);
REGISTER_CONVERTER(UnaryOnnx, Neg);
REGISTER_CONVERTER(UnaryOnnx, Sin);
REGISTER_CONVERTER(UnaryOnnx, Tan);
REGISTER_CONVERTER(UnaryOnnx, Tanh);
REGISTER_CONVERTER(UnaryOnnx, Reciprocal);
REGISTER_CONVERTER(UnaryOnnx, Round);
REGISTER_CONVERTER(UnaryOnnx, Sign);
REGISTER_CONVERTER(UnaryOnnx, Sinh);
REGISTER_CONVERTER(UnaryOnnx, Sqrt);

// For specitial error onnx
REGISTER_CONVERTER(UnaryOnnx, ATan);
