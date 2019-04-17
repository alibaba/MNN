//
//  ElementwiseTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(ElementwiseTf);

MNN::OpType ElementwiseTf::opType() {
    return MNN::OpType_Eltwise;
}
MNN::OpParameter ElementwiseTf::type() {
    return MNN::OpParameter_Eltwise;
}

void ElementwiseTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto elementwiseParam = new MNN::EltwiseT;

    if (srcNode->opType == "BiasAdd") {
        elementwiseParam->type = MNN::EltwiseType_SUM;
    }

    dstOp->main.value = elementwiseParam;
}
