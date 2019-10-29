//
//  TransposeTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(Transpose);

MNN::OpType Transpose::opType() {
    return MNN::OpType_Transpose;
}
MNN::OpParameter Transpose::type() {
    return MNN::OpParameter_Transpose;
}

void Transpose::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto Transpose = new MNN::TransposeT;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "Tperm", value)) {
        Transpose->Tperm = (MNN::DataType)value.type();
    }
    dstOp->main.value = Transpose;
}

REGISTER_CONVERTER(Transpose, Transpose);
