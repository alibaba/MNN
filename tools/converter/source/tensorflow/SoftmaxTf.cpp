//
//  SoftmaxTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(SoftmaxTf);

MNN::OpType SoftmaxTf::opType() {
    return MNN::OpType_Softmax;
}
MNN::OpParameter SoftmaxTf::type() {
    return MNN::OpParameter_Axis;
}

void SoftmaxTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto axisT  = new MNN::AxisT;
    axisT->axis = -1;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "axis", value)) {
        axisT->axis = value.i();
    }
    dstOp->main.value = axisT;
}

REGISTER_CONVERTER(SoftmaxTf, Softmax);
