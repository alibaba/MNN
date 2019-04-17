//
//  QuantizedMaxpoolTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(QuantizedMaxPool);

MNN::OpType QuantizedMaxPool::opType() {
    return MNN::OpType_QuantizedMaxPool;
}
MNN::OpParameter QuantizedMaxPool::type() {
    return MNN::OpParameter_QuantizedMaxPool;
}

void QuantizedMaxPool::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto QuantizedMaxPool = new MNN::QuantizedMaxPoolT;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "ksize", value)) {
        QuantizedMaxPool->kernelX = value.list().i(1);
        QuantizedMaxPool->kernelY = value.list().i(2);
    }

    if (find_attr_value(srcNode->tfNode, "strides", value)) {
        QuantizedMaxPool->strideX = value.list().i(1);
        QuantizedMaxPool->strideY = value.list().i(2);
    }

    if (find_attr_value(srcNode->tfNode, "padding", value)) {
        if (value.s() == "VALID") {
            QuantizedMaxPool->padType = MNN::PoolPadType_VALID;
        } else if (value.s() == "SAME") {
            QuantizedMaxPool->padType = MNN::PoolPadType_SAME;
        } else {
            LOG(ERROR) << "Not Support This Padding Mode";
        }
    }

    dstOp->main.value = QuantizedMaxPool;

    CHECK(srcNode->inEdges.size() == 1) << "QuantizedMaxPool Input ERROR";
}

REGISTER_CONVERTER(QuantizedMaxPool, QuantizedMaxPool);
