//
//  RequantizeTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(Requantize);

MNN::OpType Requantize::opType() {
    return MNN::OpType_Requantize;
}
MNN::OpParameter Requantize::type() {
    return MNN::OpParameter_Requantize;
}

void Requantize::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto Requantize = new MNN::RequantizeT;

    dstOp->main.value = Requantize;

    CHECK(srcNode->inEdges.size() == 2) << "RequantizationRange Input ERROR";
}

REGISTER_CONVERTER(Requantize, Requantize);
