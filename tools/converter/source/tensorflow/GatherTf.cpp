//
//  GatherTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(GatherTf);

MNN::OpType GatherTf::opType() {
    return MNN::OpType_Gather;
}
MNN::OpParameter GatherTf::type() {
    return MNN::OpParameter_Gather;
}

void GatherTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto parameter  = new MNN::GatherT;
    parameter->axis = 1;

    tensorflow::AttrValue value;

    find_attr_value(srcNode->tfNode, "Tindices", value);
    parameter->Tindices = (MNN::DataType)value.type();

    find_attr_value(srcNode->tfNode, "Tparams", value);
    parameter->Tparams = (MNN::DataType)value.type();

    dstOp->main.value = parameter;
}

REGISTER_CONVERTER(GatherTf, Gather);

DECLARE_OP_CONVERTER(GatherNDTf);
MNN::OpType GatherNDTf::opType() {
    return MNN::OpType_GatherND;
}
MNN::OpParameter GatherNDTf::type() {
    return MNN::OpParameter_NONE;
}
void GatherNDTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    //Do nothing
}
REGISTER_CONVERTER(GatherNDTf, GatherNd);
