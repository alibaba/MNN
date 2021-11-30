//
//  Select.cpp
//  MNNConverter
//
//  Created by MNN on 2019/05/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(SelectTf);

MNN::OpType SelectTf::opType() {
    return MNN::OpType_Select;
}
MNN::OpParameter SelectTf::type() {
    return MNN::OpParameter_NONE;
}
void SelectTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    // Do nothing
}
REGISTER_CONVERTER(SelectTf, SelectV2);
