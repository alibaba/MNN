//
//  ListDiff.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(ListDiff);

MNN::OpType ListDiff::opType() {
    return MNN::OpType_SetDiff1D;
}
MNN::OpParameter ListDiff::type() {
    return MNN::OpParameter_NONE;
}

void ListDiff::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(ListDiff, ListDiff);
REGISTER_CONVERTER(ListDiff, SetDiff1d);
