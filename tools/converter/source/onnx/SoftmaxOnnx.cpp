//
//  SoftmaxOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(SoftmaxOnnx);

MNN::OpType SoftmaxOnnx::opType() {
    return MNN::OpType_Softmax;
}
MNN::OpParameter SoftmaxOnnx::type() {
    return MNN::OpParameter_Axis;
}

void SoftmaxOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                      std::vector<const onnx::TensorProto*> initializers) {
    auto axis  = new MNN::AxisT;
    axis->axis = 1;

    dstOp->main.value = axis;
}

REGISTER_CONVERTER(SoftmaxOnnx, Softmax);
