//
//  WhereTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(WhereTf);

MNN::OpType WhereTf::opType() {
    return MNN::OpType_Where;
}
MNN::OpParameter WhereTf::type() {
    return MNN::OpParameter_Extra;
}

void WhereTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    // for distinguish old-version
    auto parameter = new MNN::ExtraT;
    parameter->engine = "tensorflow";
    parameter->type = "control_flow_where";
    dstOp->main.value = parameter;
}

REGISTER_CONVERTER(WhereTf, Where);
