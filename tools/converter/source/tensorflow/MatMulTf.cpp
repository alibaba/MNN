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

void MatMulTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
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

    dstOp->main.value = matmulParam;
}

REGISTER_CONVERTER(MatMulTf, MatMul);

DECLARE_OP_CONVERTER(MatBandPartTf);

MNN::OpType MatBandPartTf::opType() {
    return MNN::OpType_MatrixBandPart;
}
MNN::OpParameter MatBandPartTf::type() {
    return MNN::OpParameter_NONE;
}
void MatBandPartTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    //Do nothing
}

REGISTER_CONVERTER(MatBandPartTf, MatrixBandPart);
REGISTER_CONVERTER(MatBandPartTf, BatchMatrixBandPart);
