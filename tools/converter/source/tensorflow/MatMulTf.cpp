//
//  MatMulTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(MatMulTf);

MNN::OpType MatMulTf::opType() {
    return MNN::OpType_MatMul;
}
MNN::OpParameter MatMulTf::type() {
    return MNN::OpParameter_MatMul;
}

void MatMulTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto matmulParam = new MNN::MatMulT;

    tensorflow::AttrValue value;

    if (find_attr_value(srcNode->tfNode, "T", value)) {
        matmulParam->T = (MNN::DataType)value.type();
    }

    if (find_attr_value(srcNode->tfNode, "transpose_a", value)) {
        matmulParam->transposeA = value.b();
    }

    if (find_attr_value(srcNode->tfNode, "transpose_b", value)) {
        matmulParam->transposeB = value.b();
    }

    DCHECK(srcNode->inTensors.size() == 2) << "MatMul Input ERROR";
    DCHECK(srcNode->outTensors.size() == 1) << "MatMul Ouput One Tensor!!! " << srcNode->opName;

    dstOp->main.value = matmulParam;
}

REGISTER_CONVERTER(MatMulTf, MatMul);
