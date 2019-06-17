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

void BinartOpTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto parameter = new MNN::BinaryOpT;

    if (srcNode->opType == "Mul") {
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
    } else if (srcNode->opType == "GreaterEqual") {
        parameter->opType = MNN::BinaryOpOperation_GREATER_EQUAL;
    } else if (srcNode->opType == "Greater") {
        parameter->opType = MNN::BinaryOpOperation_GREATER;
    } else if (srcNode->opType == "Equal") {
        parameter->opType = MNN::BinaryOpOperation_EQUAL;
    } else if (srcNode->opType == "FloorDiv") {
        parameter->opType = MNN::BinaryOpOperation_FLOORDIV;
    } else if (srcNode->opType == "SquaredDifference") {
        parameter->opType = MNN::BinaryOpOperation_SquaredDifference;
    } else if (srcNode->opType == "Pow") {
        parameter->opType = MNN::BinaryOpOperation_POW;
    } else {
        DLOG(ERROR) << "MNN Converter Not "
                       "Supported!!!";
    }

    tensorflow::AttrValue value;
    find_attr_value(srcNode->tfNode, "T", value);
    parameter->T = (MNN::DataType)value.type();

    dstOp->main.value = parameter;
    DCHECK(srcNode->inTensors.size() == 2) << "BinaryOp Input ERROR: " << srcNode->opName << "-->" << srcNode->opType;
}

REGISTER_CONVERTER(BinartOpTf, Mul);
REGISTER_CONVERTER(BinartOpTf, Sub);
REGISTER_CONVERTER(BinartOpTf, Add);
REGISTER_CONVERTER(BinartOpTf, Maximum);
REGISTER_CONVERTER(BinartOpTf, RealDiv);
REGISTER_CONVERTER(BinartOpTf, Minimum);
REGISTER_CONVERTER(BinartOpTf, Greater);
REGISTER_CONVERTER(BinartOpTf, Equal);
REGISTER_CONVERTER(BinartOpTf, BiasAdd);
REGISTER_CONVERTER(BinartOpTf, Less);
REGISTER_CONVERTER(BinartOpTf, GreaterEqual);
REGISTER_CONVERTER(BinartOpTf, FloorDiv);
REGISTER_CONVERTER(BinartOpTf, SquaredDifference);
REGISTER_CONVERTER(BinartOpTf, Pow);
