//
//  AttentionOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2023/09/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(FmhaV2Onnx);
DECLARE_OP_CONVERTER(FmhcaOnnx);

MNN::OpType FmhaV2Onnx::opType() {
    return MNN::OpType_FmhaV2;
}

MNN::OpParameter FmhaV2Onnx::type() {
    return MNN::OpParameter_NONE;
}

void FmhaV2Onnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                      OnnxScope* scope) {
    return;
}

MNN::OpType FmhcaOnnx::opType() {
    return MNN::OpType_Fmhca;
}

MNN::OpParameter FmhcaOnnx::type() {
    return MNN::OpParameter_NONE;
}

void FmhcaOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                      OnnxScope* scope) {
    return;
}

REGISTER_CONVERTER(FmhaV2Onnx, fMHA_V2);
REGISTER_CONVERTER(FmhcaOnnx, fMHCA);