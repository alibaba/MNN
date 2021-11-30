//
//  AsStringTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(AsStringTf);

MNN::OpType AsStringTf::opType() {
    return MNN::OpType_AsString;
}
MNN::OpParameter AsStringTf::type() {
    return MNN::OpParameter_AsString;
}

void AsStringTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto parameter = new MNN::AsStringT;

    tensorflow::AttrValue value;

    find_attr_value(srcNode->tfNode, "T", value);
    parameter->T = (MNN::DataType)value.type();

    find_attr_value(srcNode->tfNode, "precision", value);
    parameter->precision = value.i();

    find_attr_value(srcNode->tfNode, "scientific", value);
    parameter->scientific = value.b();

    find_attr_value(srcNode->tfNode, "shortest", value);
    parameter->shortest = value.b();

    find_attr_value(srcNode->tfNode, "width", value);
    parameter->width = value.i();

    find_attr_value(srcNode->tfNode, "fillString", value);
    parameter->fillString = value.s();

    dstOp->main.value = parameter;
}

//REGISTER_CONVERTER(AsStringTf, AsString);
