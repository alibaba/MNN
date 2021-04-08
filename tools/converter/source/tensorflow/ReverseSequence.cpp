//
//  ReverseSequence.cpp
//  MNNConverter
//
//  Created by MNN on 2019/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(ReverseSequence);

MNN::OpType ReverseSequence::opType() {
    return MNN::OpType_ReverseSequence;
}
MNN::OpParameter ReverseSequence::type() {
    return MNN::OpParameter_ReverseSequenceParam;
}

void ReverseSequence::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto param      = new MNN::ReverseSequenceParamT;
    param->batchDim = 0;
    param->seqDim   = 0;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "batch_dim", value)) {
        param->batchDim = value.i();
    }
    if (find_attr_value(srcNode->tfNode, "seq_dim", value)) {
        param->seqDim = value.i();
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(ReverseSequence, ReverseSequence);

DECLARE_OP_CONVERTER(Reverse);

MNN::OpType Reverse::opType() {
    return MNN::OpType_Reverse;
}
MNN::OpParameter Reverse::type() {
    return MNN::OpParameter_NONE;
}

void Reverse::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(Reverse, ReverseV2);
