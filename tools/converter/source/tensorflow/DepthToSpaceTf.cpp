//
//  DepthToSpaceTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(DepthToSpaceTf);

MNN::OpType DepthToSpaceTf::opType() {
    return MNN::OpType_DepthToSpace;
}
MNN::OpParameter DepthToSpaceTf::type() {
    return MNN::OpParameter_DepthSpaceParam;
}

// input: tensor
void DepthToSpaceTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto depthToSpaceParam = new MNN::DepthSpaceParamT;
    tensorflow::AttrValue value;

    if (find_attr_value(srcNode->tfNode, "block_size", value)) {
        depthToSpaceParam->blockSize = value.i();
    } else {
        DLOG(ERROR) << "block_size not found";
    }

    dstOp->main.value = depthToSpaceParam;
}

REGISTER_CONVERTER(DepthToSpaceTf, DepthToSpace);
