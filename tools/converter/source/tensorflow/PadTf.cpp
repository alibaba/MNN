//
//  PadTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(PadTf);

MNN::OpType PadTf::opType() {
    return MNN::OpType_Padding;
}
MNN::OpParameter PadTf::type() {
    return MNN::OpParameter_NONE;
}

void PadTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    //Do nothing
}

REGISTER_CONVERTER(PadTf, Pad);
