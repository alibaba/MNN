//
//  CropAndResizeTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(CropAndResize);

MNN::OpType CropAndResize::opType() {
    return MNN::OpType_CropAndResize;
}
MNN::OpParameter CropAndResize::type() {
    return MNN::OpParameter_CropAndResize;
}

void CropAndResize::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto CropAndResize = new MNN::CropAndResizeT;
    tensorflow::AttrValue value;

    if (find_attr_value(srcNode->tfNode, "extrapolation_value", value)) {
        CropAndResize->extrapolationValue = value.f();
    }

    if (find_attr_value(srcNode->tfNode, "method", value)) {
        if (value.s() == "bilinear") {
            CropAndResize->method = MNN::CropAndResizeMethod_BILINEAR;
        } else {
            CropAndResize->method = MNN::CropAndResizeMethod_NEAREST;
        }
    }

    dstOp->main.value = CropAndResize;
}

REGISTER_CONVERTER(CropAndResize, CropAndResize);
