//
//  ShapeOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ShapeOnnx);

MNN::OpType ShapeOnnx::opType() {
    return MNN::OpType_Shape;
}
MNN::OpParameter ShapeOnnx::type() {
    return MNN::OpParameter_NONE;
}

void ShapeOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                    OnnxScope* scope) {
    dstOp->defaultDimentionFormat = MNN::MNN_DATA_FORMAT_NCHW;
}

REGISTER_CONVERTER(ShapeOnnx, Shape);

DECLARE_OP_CONVERTER(SizeOnnx);

MNN::OpType SizeOnnx::opType() {
    return MNN::OpType_Size;
}
MNN::OpParameter SizeOnnx::type() {
    return MNN::OpParameter_NONE;
}

void SizeOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                   OnnxScope* scope) {
    dstOp->defaultDimentionFormat = MNN::MNN_DATA_FORMAT_NCHW;
}

REGISTER_CONVERTER(SizeOnnx, Size);
