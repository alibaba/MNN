//
//  CosineSimilarityOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(CosineSimilarityOnnx);

MNN::OpType CosineSimilarityOnnx::opType() {
    return MNN::OpType_CosineSimilarity;
}

MNN::OpParameter CosineSimilarityOnnx::type() {
    return MNN::OpParameter_NONE;
}

void CosineSimilarityOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                               std::vector<const onnx::TensorProto *> initializers) {
    return;
}

REGISTER_CONVERTER(CosineSimilarityOnnx, ATen);
