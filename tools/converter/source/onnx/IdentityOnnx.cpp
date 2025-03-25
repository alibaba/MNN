//
//  IdentityOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(IdentityOnnx);

MNN::OpType IdentityOnnx::opType() {
    return MNN::OpType_Identity;
}
MNN::OpParameter IdentityOnnx::type() {
    return MNN::OpParameter_NONE;
}

void IdentityOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope* scope) {
    // Do nothing
    return;
}

REGISTER_CONVERTER(IdentityOnnx, Identity);


DECLARE_OP_CONVERTER(DropoutOnnx);

MNN::OpType DropoutOnnx::opType() {
    return MNN::OpType_Dropout;
}
MNN::OpParameter DropoutOnnx::type() {
    return MNN::OpParameter_NONE;
}

void DropoutOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope* scope) {
    // Do nothing
    return;
}

REGISTER_CONVERTER(DropoutOnnx, Dropout);
