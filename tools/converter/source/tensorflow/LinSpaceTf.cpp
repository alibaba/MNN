//
//  LinSpaceTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"
#include <map>
#include <string>
#include "graph.pb.h"

using namespace MNN;

DECLARE_OP_CONVERTER(LinSpaceTf);

MNN::OpType LinSpaceTf::opType() {
    return MNN::OpType_LinSpace;
}

MNN::OpParameter LinSpaceTf::type() {
    return MNN::OpParameter_NONE;
}

void LinSpaceTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(LinSpaceTf, LinSpace);
