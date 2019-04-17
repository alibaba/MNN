//
//  QuantizedAvgpoolTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(QuantizedAvgPool);

MNN::OpType QuantizedAvgPool::opType() {
    return MNN::OpType_QuantizedAvgPool;
}
MNN::OpParameter QuantizedAvgPool::type() {
    return MNN::OpParameter_QuantizedAvgPool;
}

void QuantizedAvgPool::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto QuantizedAvgPool = new MNN::QuantizedAvgPoolT;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "ksize", value)) {
        QuantizedAvgPool->kernelX = value.list().i(1);
        QuantizedAvgPool->kernelY = value.list().i(2);
    }

    if (find_attr_value(srcNode->tfNode, "strides", value)) {
        QuantizedAvgPool->strideX = value.list().i(1);
        QuantizedAvgPool->strideY = value.list().i(2);
    }

    if (find_attr_value(srcNode->tfNode, "padding", value)) {
        if (value.s() == "VALID") {
            QuantizedAvgPool->padType = MNN::PoolPadType_VALID;
        } else if (value.s() == "SAME") {
            QuantizedAvgPool->padType = MNN::PoolPadType_SAME;
        } else {
            LOG(ERROR) << "Not Support This Padding Mode";
        }
    }

    dstOp->main.value = QuantizedAvgPool;

    CHECK(srcNode->inEdges.size() == 1) << "QuantizedAvgPool Input ERROR";
}

REGISTER_CONVERTER(QuantizedAvgPool, QuantizedAvgPool);
