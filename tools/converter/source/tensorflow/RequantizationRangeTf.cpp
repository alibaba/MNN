//
//  RequantizationRangeTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(RequantizationRange);

MNN::OpType RequantizationRange::opType() {
    return MNN::OpType_RequantizationRange;
}
MNN::OpParameter RequantizationRange::type() {
    return MNN::OpParameter_RequantizationRange;
}

void RequantizationRange::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    dstOp->main.value = nullptr;
    CHECK(srcNode->inEdges.size() == 1) << "RequantizationRange Input ERROR";
}

REGISTER_CONVERTER(RequantizationRange, RequantizationRange);
