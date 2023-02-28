//
//  AddNTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"
#include <map>
#include <string>

using namespace MNN;

DECLARE_OP_CONVERTER(AddNTf);

MNN::OpType AddNTf::opType() {
    return MNN::OpType_Eltwise;
}

MNN::OpParameter AddNTf::type() {
    return MNN::OpParameter_Eltwise;
}

void AddNTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto elt          = new MNN::EltwiseT;
    dstOp->main.value = elt;
    elt->type = MNN::EltwiseType_SUM;
}

REGISTER_CONVERTER(AddNTf, AddN);
REGISTER_CONVERTER(AddNTf, AccumulateNV2);
