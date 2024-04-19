//
//  CastOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(DynamicQuantOnnx);

MNN::OpType DynamicQuantOnnx::opType() {
    return MNN::OpType_DynamicQuant;
}
MNN::OpParameter DynamicQuantOnnx::type() {
    return MNN::OpParameter_NONE;
}

void DynamicQuantOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope* scope) {
}

REGISTER_CONVERTER(DynamicQuantOnnx, DynamicQuantizeLinear);
