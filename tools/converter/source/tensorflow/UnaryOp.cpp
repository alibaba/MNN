//
//  UnaryOp.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(UnaryOpTf);

MNN::OpType UnaryOpTf::opType() {
    return MNN::OpType_UnaryOp;
}
MNN::OpParameter UnaryOpTf::type() {
    return MNN::OpParameter_UnaryOp;
}

void UnaryOpTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto parameter = new MNN::UnaryOpT;

    tensorflow::AttrValue value;

    find_attr_value(srcNode->tfNode, "T", value);
    parameter->T = (MNN::DataType)value.type();

    if (srcNode->opType == "Square") {
        parameter->opType = MNN::UnaryOpOperation_SQUARE;
    } else if (srcNode->opType == "Rsqrt") {
        parameter->opType = MNN::UnaryOpOperation_RSQRT;
    } else if (srcNode->opType == "Log1p") {
        parameter->opType = MNN::UnaryOpOperation_LOG1P;
    } else if (srcNode->opType == "Reciprocal") {
        parameter->opType = MNN::UnaryOpOperation_RECIPROCAL;
    } else if (srcNode->opType == "Exp") {
        parameter->opType = MNN::UnaryOpOperation_EXP;
    } else if (srcNode->opType == "Neg") {
        parameter->opType = MNN::UnaryOpOperation_NEG;
    } else if (srcNode->opType == "Abs") {
        parameter->opType = MNN::UnaryOpOperation_ABS;
    } else if (srcNode->opType == "Ceil") {
        parameter->opType = MNN::UnaryOpOperation_CEIL;
    } else if (srcNode->opType == "Sqrt") {
        parameter->opType = MNN::UnaryOpOperation_SQRT;
    } else if (srcNode->opType == "Log") {
        parameter->opType = MNN::UnaryOpOperation_LOG;
    } else if (srcNode->opType == "Cos") {
        parameter->opType = MNN::UnaryOpOperation_COS;
    } else if (srcNode->opType == "Tan") {
        parameter->opType = MNN::UnaryOpOperation_TAN;
    } else if (srcNode->opType == "Sin") {
        parameter->opType = MNN::UnaryOpOperation_SIN;
    } else if (srcNode->opType == "ATan") {
        parameter->opType = MNN::UnaryOpOperation_ATAN;
    } else if (srcNode->opType == "Acosh") {
        parameter->opType = MNN::UnaryOpOperation_ACOSH;
    } else if (srcNode->opType == "Sinh") {
        parameter->opType = MNN::UnaryOpOperation_SINH;
    } else if (srcNode->opType == "Asinh") {
        parameter->opType = MNN::UnaryOpOperation_ASINH;
    } else if (srcNode->opType == "Atanh") {
        parameter->opType = MNN::UnaryOpOperation_ATANH;
    } else if (srcNode->opType == "Sign") {
        parameter->opType = MNN::UnaryOpOperation_SIGN;
    } else if (srcNode->opType == "Round") {
        parameter->opType = MNN::UnaryOpOperation_ROUND;
    } else if (srcNode->opType == "Cosh") {
        parameter->opType = MNN::UnaryOpOperation_COSH;
    } else if (srcNode->opType == "Erf") {
        parameter->opType = MNN::UnaryOpOperation_ERF;
    } else if (srcNode->opType == "Erfc") {
        parameter->opType = MNN::UnaryOpOperation_ERFC;
    } else if (srcNode->opType == "Erfinv") {
        parameter->opType = MNN::UnaryOpOperation_ERFINV;
    } else if (srcNode->opType == "Expm1") {
        parameter->opType = MNN::UnaryOpOperation_EXPM1;
    } else if (srcNode->opType == "Inv") {
        parameter->opType = MNN::UnaryOpOperation_RECIPROCAL;
    } else if (srcNode->opType == "Floor") {
        parameter->opType = MNN::UnaryOpOperation_FLOOR;
    // LogicalNot is handled in tfextra
    // } else if (srcNode->opType == "LogicalNot") {
    //     parameter->opType = MNN::UnaryOpOperation_LOGICALNOT;
    } else {
        LOG(ERROR) << "MNN Converter Not "
                      "Supported!!! UnaryOp: "
                      << srcNode->opType;
    }

    dstOp->main.value = parameter;
}

REGISTER_CONVERTER(UnaryOpTf, Square);
REGISTER_CONVERTER(UnaryOpTf, Rsqrt);
REGISTER_CONVERTER(UnaryOpTf, Exp);
REGISTER_CONVERTER(UnaryOpTf, Neg);
REGISTER_CONVERTER(UnaryOpTf, Abs);
REGISTER_CONVERTER(UnaryOpTf, Ceil);
REGISTER_CONVERTER(UnaryOpTf, Sqrt);
REGISTER_CONVERTER(UnaryOpTf, Log1p);
REGISTER_CONVERTER(UnaryOpTf, Log);
REGISTER_CONVERTER(UnaryOpTf, Cos);
REGISTER_CONVERTER(UnaryOpTf, Sin);
REGISTER_CONVERTER(UnaryOpTf, ATan);
REGISTER_CONVERTER(UnaryOpTf, Tan);
REGISTER_CONVERTER(UnaryOpTf, Reciprocal);
REGISTER_CONVERTER(UnaryOpTf, Acosh);
REGISTER_CONVERTER(UnaryOpTf, Sinh);
REGISTER_CONVERTER(UnaryOpTf, Asinh);
REGISTER_CONVERTER(UnaryOpTf, Atanh);
REGISTER_CONVERTER(UnaryOpTf, Sign);
REGISTER_CONVERTER(UnaryOpTf, Round);
REGISTER_CONVERTER(UnaryOpTf, Cosh);
REGISTER_CONVERTER(UnaryOpTf, Erf);
REGISTER_CONVERTER(UnaryOpTf, Erfc);
REGISTER_CONVERTER(UnaryOpTf, Erfinv);
REGISTER_CONVERTER(UnaryOpTf, Expm1);
REGISTER_CONVERTER(UnaryOpTf, Inv);
REGISTER_CONVERTER(UnaryOpTf, Floor);
