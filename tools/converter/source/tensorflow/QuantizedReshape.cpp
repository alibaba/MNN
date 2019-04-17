//
//  QuantizedReshape.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(QuantizedReshape);

MNN::OpType QuantizedReshape::opType() {
    return MNN::OpType_QuantizedReshape;
}
MNN::OpParameter QuantizedReshape::type() {
    return MNN::OpParameter_QuantizedReshape;
}

void QuantizedReshape::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    dstOp->main.value = nullptr;

    CHECK(srcNode->inEdges.size() == 2) << "QuantizedReshape Input ERROR";
}

REGISTER_CONVERTER(QuantizedReshape, QuantizedReshape);
