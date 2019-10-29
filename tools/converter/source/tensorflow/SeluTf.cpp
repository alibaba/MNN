//
//  SeluTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(SeluTf);

MNN::OpType SeluTf::opType() {
    return MNN::OpType_Selu;
}
MNN::OpParameter SeluTf::type() {
    return MNN::OpParameter_Selu;
}

/// !!! notice ! selu parameters's value are hardcoded!!!
void SeluTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto Selu = new MNN::SeluT;

    tensorflow::AttrValue value;

    find_attr_value(srcNode->tfNode, "alpha", value);
    Selu->alpha = value.f();
    Selu->alpha = 1.6732632423543772848170429916717;

    find_attr_value(srcNode->tfNode, "scale", value);
    Selu->scale = value.f();
    Selu->scale = 1.0507009873554804934193349852946;

    dstOp->main.value = Selu;
}

REGISTER_CONVERTER(SeluTf, Selu);
