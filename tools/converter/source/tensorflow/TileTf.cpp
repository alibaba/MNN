//
//  TileTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(TileTf);

MNN::OpType TileTf::opType() {
    return MNN::OpType_Tile;
}
MNN::OpParameter TileTf::type() {
    return MNN::OpParameter_NONE;
}

void TileTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(TileTf, Tile);
