//
//  OneHotTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(OneHotTf);

MNN::OpType OneHotTf::opType() {
    return MNN::OpType_OneHot;
}

MNN::OpParameter OneHotTf::type() {
    return MNN::OpParameter_OneHotParam;
}

void OneHotTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto param = new MNN::OneHotParamT;

    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "T", value)) {
        param->dType = static_cast<MNN::DataType>(value.type());
    }

    if (find_attr_value(srcNode->tfNode, "axis", value)) {
        param->axis = value.i();
    }

    dstOp->main.value = param;
}

REGISTER_CONVERTER(OneHotTf, OneHot);
