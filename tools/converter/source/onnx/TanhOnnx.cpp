//
//  TanhOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(TanhOnnx);

MNN::OpType TanhOnnx::opType() {
    return MNN::OpType_UnaryOp;
}
MNN::OpParameter TanhOnnx::type() {
    return MNN::OpParameter_UnaryOp;
}

void TanhOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                      OnnxScope* scope) {
    auto res = new MNN::UnaryOpT;
    res->opType = MNN::UnaryOpOperation_TANH;
    dstOp->main.value = res;
}

REGISTER_CONVERTER(TanhOnnx, Tanh);
