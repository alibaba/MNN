//
//  RangeTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(Range);

MNN::OpType Range::opType() {
    return MNN::OpType_Range;
}
MNN::OpParameter Range::type() {
    return MNN::OpParameter_Range;
}

void Range::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto Range = new MNN::RangeT;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "Tidx", value)) {
        Range->Tidx = (MNN::DataType)value.type();
    }
    dstOp->main.value = Range;
}

REGISTER_CONVERTER(Range, Range);
