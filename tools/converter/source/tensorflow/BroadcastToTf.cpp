//
//  BroadcastToTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(BroadcastToTf);

MNN::OpType BroadcastToTf::opType() {
    return MNN::OpType_BroadcastTo;
}
MNN::OpParameter BroadcastToTf::type() {
    return MNN::OpParameter_NONE;
}

void BroadcastToTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(BroadcastToTf, BroadcastTo);
