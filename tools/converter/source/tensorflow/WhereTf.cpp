//
//  WhereTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(WhereTf);

MNN::OpType WhereTf::opType() {
    return MNN::OpType_Where;
}
MNN::OpParameter WhereTf::type() {
    return MNN::OpParameter_NONE;
}

void WhereTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(WhereTf, Where);
