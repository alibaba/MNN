//
//  FillTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(FillTf);

MNN::OpType FillTf::opType() {
    return MNN::OpType_Fill;
}
MNN::OpParameter FillTf::type() {
    return MNN::OpParameter_Fill;
}

void FillTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    dstOp->main.value = nullptr;
}
REGISTER_CONVERTER(FillTf, Fill);

DECLARE_OP_CONVERTER(ZerosLikeTf);
MNN::OpType ZerosLikeTf::opType() {
    return MNN::OpType_ZerosLike;
}
MNN::OpParameter ZerosLikeTf::type() {
    return MNN::OpParameter_NONE;
}

void ZerosLikeTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(ZerosLikeTf, ZerosLike);
