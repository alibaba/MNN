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
    return MNN::OpType_Sigmoid;
}

MNN::OpParameter SigmoidOnnx::type() {
    return MNN::OpParameter_NONE;
}

void SigmoidOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                      std::vector<const onnx::TensorProto *> initializers) {
    return;
}

REGISTER_CONVERTER(SigmoidOnnx, Sigmoid);
