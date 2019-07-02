//
//  ClipOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "OnnxUtils.hpp"
#include "logkit.h"
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ClipOnnx);

MNN::OpType ClipOnnx::opType() {
    return MNN::OpType_ReLU6;
}
MNN::OpParameter ClipOnnx::type() {
    return MNN::OpParameter_NONE;
}

void ClipOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                   std::vector<const onnx::TensorProto*> initializers) {
    float maxValue = 0.0f;
    float minValue = 1.0f;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();
        if (attributeName == "max") {
            maxValue = attributeProto.f();
        }
        if (attributeName == "min") {
            minValue = attributeProto.f();
        }
    }
    DCHECK_EQ(maxValue, 6.0f);
    DCHECK_EQ(minValue, 0.0f);
}

REGISTER_CONVERTER(ClipOnnx, Clip);
