//
//  PoolingTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(PoolingTf);

MNN::OpType PoolingTf::opType() {
    return MNN::OpType_Pooling;
}
MNN::OpParameter PoolingTf::type() {
    return MNN::OpParameter_Pool;
}

// input: tensor
void PoolingTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto pool = new MNN::PoolT;

    tensorflow::AttrValue value;

    int kernel_size_h = 1;
    int kernel_size_w = 1;
    int stride_h      = 1;
    int stride_w      = 1;

    if (srcNode->opType == "AvgPool") {
        pool->type = MNN::PoolType_AVEPOOL;
    } else if (srcNode->opType == "MaxPool") {
        pool->type = MNN::PoolType_MAXPOOL;
    } else {
        DLOG(ERROR) << "Not Support This Pooling Type: " << srcNode->opType;
    }

    if (find_attr_value(srcNode->tfNode, "ksize", value)) {
        kernel_size_h = value.list().i(1);
        kernel_size_w = value.list().i(2);
    }
    pool->kernelX = kernel_size_w;
    pool->kernelY = kernel_size_h;

    if (find_attr_value(srcNode->tfNode, "strides", value)) {
        stride_h = value.list().i(1);
        stride_w = value.list().i(2);
    }
    pool->strideX = stride_w;
    pool->strideY = stride_h;

    if (find_attr_value(srcNode->tfNode, "padding", value)) {
        if (value.s() == "VALID") {
            pool->padType = MNN::PoolPadType_VALID;
        } else if (value.s() == "SAME") {
            pool->padType = MNN::PoolPadType_SAME;
        } else {
            DLOG(ERROR) << "Not Support This Padding Mode";
        }
    }
    pool->padY = 0; // runtime compute this pad
    pool->padX = 0;

    pool->isGlobal = false; // TODO

    dstOp->main.value = pool;
}

REGISTER_CONVERTER(PoolingTf, MaxPool);
REGISTER_CONVERTER(PoolingTf, AvgPool);
