//
//  ConcatTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(ConcatTf);

MNN::OpType ConcatTf::opType() {
    return MNN::OpType_Concat;
}
MNN::OpParameter ConcatTf::type() {
    return MNN::OpParameter_Axis;
}

void ConcatTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto concat = new MNN::AxisT;

    tensorflow::AttrValue value;
    // last input is axis(Const)
    TmpNode *constAxisNode = nullptr;
    if (srcNode->opType == "ConcatV2") {
        constAxisNode = tempGraph->_getTmpNode(srcNode->inEdges.back());
    } else {
        constAxisNode = tempGraph->_getTmpNode(srcNode->inEdges[0]);
    }
    if (find_attr_value(constAxisNode->tfNode, "value", value)) {
        const tensorflow::TensorProto tensor = value.tensor();
        DCHECK((MNN::DataType)tensor.dtype() == MNN::DataType_DT_INT32) << "Concat Input Const Axis Node "
                                                                           "ERROR!!! ===> "
                                                                        << srcNode->opName;
        const int axis = tensor.int_val().data()[0];
        concat->axis   = axis;
    }

    dstOp->main.value = concat;
}

REGISTER_CONVERTER(ConcatTf, ConcatV2);
REGISTER_CONVERTER(ConcatTf, Concat);
