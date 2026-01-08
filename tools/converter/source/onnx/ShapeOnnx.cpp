//
//  ShapeOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <limits>
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
    bool hasStart = false;
    bool hasEnd = false;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attribute = onnxNode->attribute(i);
        if (attribute.name() == "start") {
            hasStart = true;
        } else if (attribute.name() == "end") {
            hasEnd = true;
        }
    }

    if (hasStart || hasEnd) {
        auto shapeParam = new MNN::ShapeParamT;
        shapeParam->start = 0;
        shapeParam->end = std::numeric_limits<int>::max();

        for (int i = 0; i < onnxNode->attribute_size(); ++i) {
            const auto& attribute = onnxNode->attribute(i);
            if (attribute.name() == "start") {
                shapeParam->start = attribute.i();
            } else if (attribute.name() == "end") {
                shapeParam->end = attribute.i();
            }
        }

        dstOp->main.type = MNN::OpParameter_ShapeParam;
        dstOp->main.value = shapeParam;
    }
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
