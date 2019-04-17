//
//  Relu6Tf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(Relu6Tf);

MNN::OpType Relu6Tf::opType() {
    return MNN::OpType_ReLU6;
}
MNN::OpParameter Relu6Tf::type() {
    return MNN::OpParameter_Relu6;
}

void Relu6Tf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto relu6   = new MNN::Relu6T;
    relu6->slope = 0.0f;

    dstOp->main.value = relu6;
    DCHECK(srcNode->inTensors.size() == 1) << "Relu6 Input ERROR";
}

REGISTER_CONVERTER(Relu6Tf, Relu6);
