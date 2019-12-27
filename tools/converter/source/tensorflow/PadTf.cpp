//
//  PadTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(PadTf);

MNN::OpType PadTf::opType() {
    return MNN::OpType_Padding;
}
MNN::OpParameter PadTf::type() {
    return MNN::OpParameter_PadParam;
}

void PadTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto padparm = new MNN::PadParamT;

    padparm->mode = MNN::PadValueMode_CONSTANT;
    if (srcNode->opType == "MirrorPad") {
        tensorflow::AttrValue value;
        if (find_attr_value(srcNode->tfNode, "mode", value)) {
            if (value.s() == "SYMMETRIC") {
                padparm->mode = MNN::PadValueMode_SYMMETRIC;
            } else if (value.s() == "REFLECT") {
                padparm->mode = MNN::PadValueMode_REFLECT;
            }
        }
    }

    dstOp->main.value = padparm;
}

REGISTER_CONVERTER(PadTf, Pad);
REGISTER_CONVERTER(PadTf, PadV2);
REGISTER_CONVERTER(PadTf, MirrorPad);
