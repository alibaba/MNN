//
//  UnravelIndexTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(UnravelIndexTf);

MNN::OpType UnravelIndexTf::opType() {
    return MNN::OpType_UnravelIndex;
}

MNN::OpParameter UnravelIndexTf::type() {
    return MNN::OpParameter_NONE;
}

void UnravelIndexTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    return;
}

REGISTER_CONVERTER(UnravelIndexTf, UnravelIndex);
