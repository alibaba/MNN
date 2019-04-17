//
//  QuantizedMatMulTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(QuantizedMatMul);

MNN::OpType QuantizedMatMul::opType() {
    return MNN::OpType_QuantizedMatMul;
}
MNN::OpParameter QuantizedMatMul::type() {
    return MNN::OpParameter_QuantizedMatMul;
}

// input: tensor, weight, (no bias now)
void QuantizedMatMul::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto QuantizedMatMul = new MNN::QuantizedMatMulT;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "transpose_a", value)) {
        QuantizedMatMul->transposeA = value.b();
    }
    if (find_attr_value(srcNode->tfNode, "transpose_b", value)) {
        QuantizedMatMul->transposeB = value.b();
    }
    dstOp->main.value = QuantizedMatMul;

    CHECK(srcNode->inEdges.size() == 4) << "QuantizedMatMul Input ERROR";
}

REGISTER_CONVERTER(QuantizedMatMul, QuantizedMatMul);
