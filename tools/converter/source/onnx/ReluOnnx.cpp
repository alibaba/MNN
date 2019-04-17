//
//  ReluOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ReluOnnx);

MNN::OpType ReluOnnx::opType() {
    return MNN::OpType_ReLU;
}
MNN::OpParameter ReluOnnx::type() {
    return MNN::OpParameter_Relu;
}

void ReluOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                   std::vector<const onnx::TensorProto*> initializers) {
    auto relu         = new MNN::ReluT;
    relu->slope       = .0f;
    dstOp->main.value = relu;
}

REGISTER_CONVERTER(ReluOnnx, Relu);
