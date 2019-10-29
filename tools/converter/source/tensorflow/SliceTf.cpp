//
//  SliceTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(SliceTf);

MNN::OpType SliceTf::opType() {
    return MNN::OpType_SliceTf;
}
MNN::OpParameter SliceTf::type() {
    return MNN::OpParameter_SliceTf;
}

void SliceTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto sliceParam = new MNN::SliceTfT;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "T", value)) {
        sliceParam->T = (MNN::DataType)value.type();
    }
    dstOp->main.value = sliceParam;
}

REGISTER_CONVERTER(SliceTf, Slice);
