//
//  CastLikeOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2024/10/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(CastLikeOnnx);

MNN::OpType CastLikeOnnx::opType() {
    return MNN::OpType_CastLike;
}

MNN::OpParameter CastLikeOnnx::type() {
    return MNN::OpParameter_NONE;
}

void CastLikeOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope* scope) {
    return;
}

REGISTER_CONVERTER(CastLikeOnnx, CastLike);
