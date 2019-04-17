//
//  ReluTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(ReluTf);

MNN::OpType ReluTf::opType() {
    return MNN::OpType_ReLU;
}
MNN::OpParameter ReluTf::type() {
    return MNN::OpParameter_Relu;
}

void ReluTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto Relu   = new MNN::ReluT;
    Relu->slope = 0.0f;

    dstOp->main.value = Relu;
    DCHECK(srcNode->inTensors.size() == 1) << "Relu Input ERROR";
}

REGISTER_CONVERTER(ReluTf, Relu);
