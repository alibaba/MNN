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
    return MNN::OpType_TanH;
}
MNN::OpParameter TanhOnnx::type() {
    return MNN::OpParameter_NONE;
}

void TanhOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                      std::vector<const onnx::TensorProto*> initializers) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(TanhOnnx, Tanh);
