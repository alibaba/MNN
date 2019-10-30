//
//  NonMaxSuppressionV2Tf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(NonMaxSuppressionV2);

MNN::OpType NonMaxSuppressionV2::opType() {
    return MNN::OpType_NonMaxSuppressionV2;
}
MNN::OpParameter NonMaxSuppressionV2::type() {
    return MNN::OpParameter_NonMaxSuppressionV2;
}

void NonMaxSuppressionV2::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(NonMaxSuppressionV2, NonMaxSuppressionV2);
REGISTER_CONVERTER(NonMaxSuppressionV2, NonMaxSuppressionV3);
