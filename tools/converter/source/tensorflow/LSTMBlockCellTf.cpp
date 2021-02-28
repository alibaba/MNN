//
//  LSTMBlockCellTf.cpp
//  MNNConverter
//
//  Created by MNN on 2021/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(LSTMBlockCellTf);

MNN::OpType LSTMBlockCellTf::opType() {
    return MNN::OpType_LSTMBlockCell;
}
MNN::OpParameter LSTMBlockCellTf::type() {
    return MNN::OpParameter_LSTMBlockCell;
}

void LSTMBlockCellTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto lstmParam = new MNN::LSTMBlockCellT;
    tensorflow::AttrValue value;

    if (find_attr_value(srcNode->tfNode, "cell_clip", value)) {
        lstmParam->cell_clip = value.f();
    }

    if (find_attr_value(srcNode->tfNode, "forget_bias", value)) {
        lstmParam->forget_bias = value.f();
    }

    if (find_attr_value(srcNode->tfNode, "use_peephole", value)) {
        lstmParam->use_peephole = value.b();
    }
    dstOp->main.value = lstmParam;
}

REGISTER_CONVERTER(LSTMBlockCellTf, LSTMBlockCell);
