//
//  SigmoidOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(SigmoidOnnx);

MNN::OpType SigmoidOnnx::opType() {
    return MNN::OpType_UnaryOp;
}

MNN::OpParameter SigmoidOnnx::type() {
    return MNN::OpParameter_UnaryOp;
}

void SigmoidOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                      OnnxScope* scope) {
    auto res = new MNN::UnaryOpT;
    res->opType = MNN::UnaryOpOperation_SIGMOID;;
    dstOp->main.value = res;
    return;
}

REGISTER_CONVERTER(SigmoidOnnx, Sigmoid);
