//
//  RangeOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(RangeOnnx);

MNN::OpType RangeOnnx::opType() {
    return MNN::OpType_Range;
}

MNN::OpParameter RangeOnnx::type() {
    return MNN::OpParameter_NONE;
}

void RangeOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope* scope) {
    return;
}

REGISTER_CONVERTER(RangeOnnx, Range);
