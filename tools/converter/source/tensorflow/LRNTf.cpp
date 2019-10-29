//
//  LRNTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(LRNTf);

MNN::OpType LRNTf::opType() {
    return MNN::OpType_LRN;
}
MNN::OpParameter LRNTf::type() {
    return MNN::OpParameter_LRN;
}

void LRNTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto lrnParam        = new MNN::LRNT;
    lrnParam->regionType = 0;
    tensorflow::AttrValue value;

    find_attr_value(srcNode->tfNode, "alpha", value);
    lrnParam->alpha = value.f();

    find_attr_value(srcNode->tfNode, "beta", value);
    lrnParam->beta = value.f();

    find_attr_value(srcNode->tfNode, "depth_radius", value);
    lrnParam->localSize = 2 * value.i() + 1;

    dstOp->main.value = lrnParam;
}

REGISTER_CONVERTER(LRNTf, LRN);
