//
//  Shape.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(ShapeTf);

MNN::OpType ShapeTf::opType() {
    return MNN::OpType_Shape;
}
MNN::OpParameter ShapeTf::type() {
    return MNN::OpParameter_NONE;
}

void ShapeTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(ShapeTf, Shape);
