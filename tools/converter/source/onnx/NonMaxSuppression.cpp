//
//  NonMaxSuppression.cpp
//  MNN
//
//  Created by MNN on 2019/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(NonMaxSuppressionOnnx);


MNN::OpType NonMaxSuppressionOnnx::opType() {
    return MNN::OpType_NonMaxSuppressionV2;
}
MNN::OpParameter NonMaxSuppressionOnnx::type() {
    return MNN::OpParameter_NonMaxSuppressionV2;
}

void NonMaxSuppressionOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                  std::vector<const onnx::TensorProto*> initializers) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(NonMaxSuppressionOnnx, NonMaxSuppression);
