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

void UnaryOpTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
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
REGISTER_CONVERTER(UnaryOpTf, Reciprocal);
