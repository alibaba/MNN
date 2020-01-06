//
//  ScatterNDOnnx.cpp
//  MNN
//
//  Created by MNN on 2019/06/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ScatterNDOnnx);

MNN::OpType ScatterNDOnnx::opType() {
    return MNN::OpType_ScatterNd;
}
MNN::OpParameter ScatterNDOnnx::type() {
    return MNN::OpParameter_NONE;
}

void ScatterNDOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                      std::vector<const onnx::TensorProto*> initializers) {
    dstOp->main.value = nullptr;
}


REGISTER_CONVERTER(ScatterNDOnnx, ScatterND);
