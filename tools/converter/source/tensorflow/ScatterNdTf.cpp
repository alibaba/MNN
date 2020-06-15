//
//  ScatterNdTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(ScatterNdTf);

MNN::OpType ScatterNdTf::opType() {
    return MNN::OpType_ScatterNd;
}
MNN::OpParameter ScatterNdTf::type() {
    return MNN::OpParameter_NONE;
}

void ScatterNdTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(ScatterNdTf, ScatterNd);
