//
//  SpaceToDepth.cpp
//  MNNConverter
//
//  Created by MNN on 2019/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(SpaceToDepthTf);

MNN::OpType SpaceToDepthTf::opType() {
    return MNN::OpType_SpaceToDepth;
}
MNN::OpParameter SpaceToDepthTf::type() {
    return MNN::OpParameter_DepthSpaceParam;
}

// input: tensor
void SpaceToDepthTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto spaceToDepthParam = new MNN::DepthSpaceParamT;
    tensorflow::AttrValue value;

    if (find_attr_value(srcNode->tfNode, "block_size", value)) {
        spaceToDepthParam->blockSize = value.i();
    } else {
        DLOG(ERROR) << "block_size not found";
    }

    dstOp->main.value = spaceToDepthParam;
}

REGISTER_CONVERTER(SpaceToDepthTf, SpaceToDepth);
