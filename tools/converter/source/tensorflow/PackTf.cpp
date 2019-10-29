//
//  PackTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(PackTf);

MNN::OpType PackTf::opType() {
    return MNN::OpType_Pack;
}
MNN::OpParameter PackTf::type() {
    return MNN::OpParameter_PackParam;
}

void PackTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto pack = new MNN::PackParamT;

    tensorflow::AttrValue value;

    find_attr_value(srcNode->tfNode, "T", value);
    MNN::DataType dataType = (MNN::DataType)value.type();
    pack->dataType         = dataType;

    find_attr_value(srcNode->tfNode, "axis", value);
    pack->axis = value.i();

    dstOp->main.value = pack;
}

REGISTER_CONVERTER(PackTf, Pack);
