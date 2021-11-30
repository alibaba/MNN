//
//  ReduceJoinTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(ReduceJoinTf);

MNN::OpType ReduceJoinTf::opType() {
    return MNN::OpType_ReduceJoin;
}
MNN::OpParameter ReduceJoinTf::type() {
    return MNN::OpParameter_ReduceJoin;
}

void ReduceJoinTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto parameter = new MNN::ReduceJoinT;

    tensorflow::AttrValue value;

    find_attr_value(srcNode->tfNode, "keep_dims", value);
    parameter->keepDims = value.b();

    find_attr_value(srcNode->tfNode, "separator", value);
    parameter->separator = value.s();

    dstOp->main.value = parameter;
}

//REGISTER_CONVERTER(ReduceJoinTf, ReduceJoin);
