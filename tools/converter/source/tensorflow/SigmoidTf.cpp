//
//  SigmoidTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(SigmoidTf);

MNN::OpType SigmoidTf::opType() {
    return MNN::OpType_Sigmoid;
}
MNN::OpParameter SigmoidTf::type() {
    return MNN::OpParameter_NONE;
}

void SigmoidTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(SigmoidTf, Sigmoid);
