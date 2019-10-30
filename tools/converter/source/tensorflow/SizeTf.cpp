//
//  SizeTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(Size);

MNN::OpType Size::opType() {
    return MNN::OpType_Size;
}
MNN::OpParameter Size::type() {
    return MNN::OpParameter_Size;
}

void Size::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto Size = new MNN::SizeT;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "out_type", value)) {
        Size->outputDataType = (MNN::DataType)value.type();
    }
    dstOp->main.value = Size;
}

REGISTER_CONVERTER(Size, Size);
