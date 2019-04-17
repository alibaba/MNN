//
//  ArgMaxTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(ArgMaxTf);

MNN::OpType ArgMaxTf::opType() {
    return MNN::OpType_ArgMax;
}
MNN::OpParameter ArgMaxTf::type() {
    return MNN::OpParameter_ArgMax;
}

void ArgMaxTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto argmaxParam = new MNN::ArgMaxT;

    argmaxParam->axis             = 0;
    argmaxParam->outMaxVal        = false;
    argmaxParam->softmaxThreshold = false;
    argmaxParam->topK             = 1;

    dstOp->main.value = argmaxParam;
}

REGISTER_CONVERTER(ArgMaxTf, ArgMax);
