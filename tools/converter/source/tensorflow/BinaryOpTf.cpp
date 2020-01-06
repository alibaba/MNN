//
//  BinaryOpTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(BinartOpTf);

MNN::OpType BinartOpTf::opType() {
    return MNN::OpType_BinaryOp;
}
MNN::OpParameter BinartOpTf::type() {
    return MNN::OpParameter_BinaryOp;
}

void BinartOpTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto parameter = new MNN::BinaryOpT;

    if (srcNode->opType == "Mul" || srcNode->opType == "LogicalAnd") {
        parameter->opType = MNN::BinaryOpOperation_MUL;
    } else if (srcNode->opType == "Sub") {
        parameter->opType = MNN::BinaryOpOperation_SUB;
    } else if (srcNode->opType == "Add" || srcNode->opType == "BiasAdd") {
        parameter->opType = MNN::BinaryOpOperation_ADD;
    } else if (srcNode->opType == "RealDiv") {
        parameter->opType = MNN::BinaryOpOperation_REALDIV;
    } else if (srcNode->opType == "Maximum") {
        parameter->opType = MNN::BinaryOpOperation_MAXIMUM;
    } else if (srcNode->opType == "Minimum") {
        parameter->opType = MNN::BinaryOpOperation_MINIMUM;
    } else if (srcNode->opType == "Less") {
        parameter->opType = MNN::BinaryOpOperation_LESS;
    } else if (srcNode->opType == "LessEqual") {
        parameter->opType = MNN::BinaryOpOperation_LESS_EQUAL;
    } else if (srcNode->opType == "GreaterEqual") {
        parameter->opType = MNN::BinaryOpOperation_GREATER_EQUAL;
    } else if (srcNode->opType == "Greater") {
        parameter->opType = MNN::BinaryOpOperation_GREATER;
    } else if (srcNode->opType == "Equal") {
        parameter->opType = MNN::BinaryOpOperation_EQUAL;
    } else if (srcNode->opType == "FloorDiv") {
        parameter->opType = MNN::BinaryOpOperation_FLOORDIV;
    } else if (srcNode->opType == "FloorMod") {
        parameter->opType = MNN::BinaryOpOperation_FLOORMOD;
    } else if (srcNode->opType == "SquaredDifference") {
        parameter->opType = MNN::BinaryOpOperation_SquaredDifference;
    } else if (srcNode->opType == "Pow") {
        parameter->opType = MNN::BinaryOpOperation_POW;
    } else if (srcNode->opType == "AddV2") {
        parameter->opType = MNN::BinaryOpOperation_ADD;
    } else if (srcNode->opType == "Atan2") {
        parameter->opType = MNN::BinaryOpOperation_ATAN2;
    } else if (srcNode->opType == "LogicalOr") {
        parameter->opType = MNN::BinaryOpOperation_LOGICALOR;
    } else if (srcNode->opType == "NotEqual") {
        parameter->opType = MNN::BinaryOpOperation_NOTEQUAL;
    } else if (srcNode->opType == "TruncateDiv") {
        parameter->opType = MNN::BinaryOpOperation_REALDIV;
    } else if (srcNode->opType == "Mod") {
        parameter->opType = MNN::BinaryOpOperation_MOD;
    } else {
        DLOG(ERROR) << "MNN Converter Not "
                       "Supported!!!";
    }

    tensorflow::AttrValue value;
    find_attr_value(srcNode->tfNode, "T", value);
    parameter->T = (MNN::DataType)value.type();

    dstOp->main.value = parameter;
}

REGISTER_CONVERTER(BinartOpTf, Mul);
REGISTER_CONVERTER(BinartOpTf, LogicalAnd);
REGISTER_CONVERTER(BinartOpTf, Sub);
REGISTER_CONVERTER(BinartOpTf, Add);
REGISTER_CONVERTER(BinartOpTf, Maximum);
REGISTER_CONVERTER(BinartOpTf, RealDiv);
REGISTER_CONVERTER(BinartOpTf, Minimum);
REGISTER_CONVERTER(BinartOpTf, Greater);
REGISTER_CONVERTER(BinartOpTf, Equal);
REGISTER_CONVERTER(BinartOpTf, BiasAdd);
REGISTER_CONVERTER(BinartOpTf, Less);
REGISTER_CONVERTER(BinartOpTf, LessEqual);
REGISTER_CONVERTER(BinartOpTf, GreaterEqual);
REGISTER_CONVERTER(BinartOpTf, FloorDiv);
REGISTER_CONVERTER(BinartOpTf, FloorMod);
REGISTER_CONVERTER(BinartOpTf, SquaredDifference);
REGISTER_CONVERTER(BinartOpTf, Pow);
REGISTER_CONVERTER(BinartOpTf, AddV2);
REGISTER_CONVERTER(BinartOpTf, Atan2);
REGISTER_CONVERTER(BinartOpTf, LogicalOr);
REGISTER_CONVERTER(BinartOpTf, NotEqual);
REGISTER_CONVERTER(BinartOpTf, TruncateDiv);
REGISTER_CONVERTER(BinartOpTf, Mod);
