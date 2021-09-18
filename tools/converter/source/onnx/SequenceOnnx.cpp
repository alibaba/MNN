//
//  SplitToSequenceOnnx.cpp
//  MNN
//
//  Created by MNN on 2019/06/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(SplitToSequenceOnnx);

MNN::OpType SplitToSequenceOnnx::opType() {
    return MNN::OpType_TensorArraySplit;
}
MNN::OpParameter SplitToSequenceOnnx::type() {
    return MNN::OpParameter_TensorArray;
}

void SplitToSequenceOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                              OnnxScope* scope) {
    auto tensorArray = new MNN::TensorArrayT;
    tensorArray->T = MNN::DataType_DT_FLOAT;
    dstOp->main.value = tensorArray;
    auto tensorArrayIdx = scope->buildTensorArrayOp({}, false, dstOp->name + "/tensorArray");
    int valueIdx = dstOp->inputIndexes[0];
    int splitIdx = dstOp->inputIndexes[1];
    dstOp->inputIndexes.resize(4);
    // handle, value, lengths, flow_in
    dstOp->inputIndexes[0] = tensorArrayIdx.first;
    dstOp->inputIndexes[1] = valueIdx;
    dstOp->inputIndexes[2] = splitIdx;
    dstOp->inputIndexes[3] = tensorArrayIdx.second;
}

REGISTER_CONVERTER(SplitToSequenceOnnx, SplitToSequence);

DECLARE_OP_CONVERTER(SequenceAtOnnx);

MNN::OpType SequenceAtOnnx::opType() {
    return MNN::OpType_TensorArrayRead;
}
MNN::OpParameter SequenceAtOnnx::type() {
    return MNN::OpParameter_TensorArray;
}

void SequenceAtOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                         OnnxScope* scope) {
    auto tensorArray = new MNN::TensorArrayT;
    tensorArray->T = MNN::DataType_DT_FLOAT;
    dstOp->main.value = tensorArray;
    // handle, index, flow_in and handle == flow_in
    dstOp->inputIndexes.push_back(dstOp->inputIndexes[0]);
}

REGISTER_CONVERTER(SequenceAtOnnx, SequenceAt);
