//
//  EluTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(EluTf);

MNN::OpType EluTf::opType() {
    return MNN::OpType_ELU;
}
MNN::OpParameter EluTf::type() {
    return MNN::OpParameter_ELU;
}

void EluTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto elu = new MNN::ELUT;
    elu->alpha = 1.0f;
    dstOp->main.value = elu;
}

REGISTER_CONVERTER(EluTf, Elu);
