//
//  ConstTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"
#include <map>
#include <string>
#include "graph.pb.h"
using namespace MNN;
DECLARE_OP_CONVERTER(ConstTf);

MNN::OpType ConstTf::opType() {
    return MNN::OpType_Const;
}
MNN::OpParameter ConstTf::type() {
    return MNN::OpParameter_Blob;
}


void ConstTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto parameter = new MNN::BlobT;
    tensorflow::AttrValue weightsValue;
    if (!find_attr_value(srcNode->tfNode, "value", weightsValue)) {
        LOG(ERROR) << "Const Node Have Not Data!!!==> " << srcNode->opName;
    }
    tfOpConverter::convertTensorToBlob(parameter, weightsValue.tensor());
    dstOp->main.value = parameter;
    // CHECK(srcNode->inTensors.size() == 0) << "Const Should Not Have Input!!! ===> " << srcNode->opName;
}

REGISTER_CONVERTER(ConstTf, Const);
REGISTER_CONVERTER(ConstTf, HostConst);
