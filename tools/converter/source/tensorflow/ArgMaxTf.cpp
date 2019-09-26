//
//  ArgMaxTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(ArgMaxTf);

MNN::OpType ArgMaxTf::opType() {
    return MNN::OpType_ArgMax;
}
MNN::OpParameter ArgMaxTf::type() {
    return MNN::OpParameter_ArgMax;
}

void ArgMaxTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto argmaxParam = new MNN::ArgMaxT;

    argmaxParam->outMaxVal        = false;
    argmaxParam->softmaxThreshold = false;
    argmaxParam->topK             = 1;

    int axis = 0;
    if (srcNode->inEdges.size() == 2) {
        // axis from input Constant node
        auto axisNode = tempGraph->_getTmpNode(srcNode->inEdges[1]);
        DCHECK(axisNode->opType == "Const") << "The Second Input of Argmax should be Constant";
        tensorflow::AttrValue value;
        if (find_attr_value(axisNode->tfNode, "value", value)) {
            DCHECK(value.tensor().int_val_size() == 1) << "input axis error!";
            axis = value.tensor().int_val(0);
        }
    }

    argmaxParam->axis = axis;
    dstOp->main.value = argmaxParam;
}

REGISTER_CONVERTER(ArgMaxTf, ArgMax);
