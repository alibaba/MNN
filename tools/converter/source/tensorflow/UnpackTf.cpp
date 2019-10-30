//
//  UnpackTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(UnpackTf);

MNN::OpType UnpackTf::opType() {
    return MNN::OpType_Unpack;
}
MNN::OpParameter UnpackTf::type() {
    return MNN::OpParameter_Axis;
}

void UnpackTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto axisT = new MNN::AxisT;
    tensorflow::AttrValue value;
    axisT->axis = 1; // default
    find_attr_value(srcNode->tfNode, "axis", value);
    axisT->axis       = value.i();
    dstOp->main.value = axisT;
}

REGISTER_CONVERTER(UnpackTf, Unpack);
