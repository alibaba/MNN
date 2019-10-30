//
//  GatherV2Tf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(GatherV2);

MNN::OpType GatherV2::opType() {
    return MNN::OpType_GatherV2;
}
MNN::OpParameter GatherV2::type() {
    return MNN::OpParameter_GatherV2;
}

void GatherV2::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto GatherV2 = new MNN::GatherV2T;
    tensorflow::AttrValue value;

    find_attr_value(srcNode->tfNode, "Taxis", value);
    GatherV2->Taxis = (MNN::DataType)value.type();

    find_attr_value(srcNode->tfNode, "Tindices", value);
    GatherV2->Tindices = (MNN::DataType)value.type();

    find_attr_value(srcNode->tfNode, "Tparams", value);
    GatherV2->Tparams = (MNN::DataType)value.type();

    dstOp->main.value = GatherV2;
}

REGISTER_CONVERTER(GatherV2, GatherV2);
