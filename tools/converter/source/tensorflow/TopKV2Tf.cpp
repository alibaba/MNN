//
//  TopKV2Tf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(TopKV2Tf);

MNN::OpType TopKV2Tf::opType() {
    return MNN::OpType_TopKV2;
}
MNN::OpParameter TopKV2Tf::type() {
    return MNN::OpParameter_TopKV2;
}

void TopKV2Tf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto topkv2Param = new MNN::TopKV2T;

    tensorflow::AttrValue value;
    topkv2Param->sorted = false;
    if (find_attr_value(srcNode->tfNode, "sorted", value)) {
        topkv2Param->sorted = value.b();
    }

    topkv2Param->T = MNN::DataType_DT_FLOAT;
    if (find_attr_value(srcNode->tfNode, "T", value)) {
        topkv2Param->T = (MNN::DataType)value.type();
    }
    dstOp->outputIndexes = {-1, -1};

    dstOp->main.value = topkv2Param;
}

REGISTER_CONVERTER(TopKV2Tf, TopKV2);
