//
//  ReluOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ReluOnnx);

MNN::OpType ReluOnnx::opType() {
    return MNN::OpType_ReLU;
}
MNN::OpParameter ReluOnnx::type() {
    return MNN::OpParameter_Relu;
}

void ReluOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                   OnnxScope* scope) {
    auto relu = new MNN::ReluT;

    float slope         = 0.01f;
    const auto attrSize = onnxNode->attribute_size();
    for (int i = 0; i < attrSize; ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();

        if (attributeName == "alpha") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_FLOAT) << "Node Attribute ERROR";
            slope = attributeProto.f();
        } else {
            DLOG(ERROR) << "TODO!";
        }
    }

    if (onnxNode->op_type() == "LeakyRelu") {
        relu->slope = slope;
    } else {
        relu->slope = .0f;
    }

    dstOp->main.value = relu;
}

REGISTER_CONVERTER(ReluOnnx, Relu);
REGISTER_CONVERTER(ReluOnnx, LeakyRelu);
