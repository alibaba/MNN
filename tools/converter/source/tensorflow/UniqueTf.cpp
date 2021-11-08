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

DECLARE_OP_CONVERTER(Unique);

MNN::OpType Unique::opType() {
    return MNN::OpType_Unique;
}
MNN::OpParameter Unique::type() {
    return MNN::OpParameter_NONE;
}

void Unique::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(Unique, Unique);
