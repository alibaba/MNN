//
//  RankTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(Rank);

MNN::OpType Rank::opType() {
    return MNN::OpType_Rank;
}
MNN::OpParameter Rank::type() {
    return MNN::OpParameter_NONE;
}

void Rank::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(Rank, Rank);
