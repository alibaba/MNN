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
                   std::vector<const onnx::TensorProto*> initializers) {
    auto relu = new MNN::ReluT;

    float slope         = 0.01;
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

DECLARE_OP_CONVERTER(PReluOnnx);

MNN::OpType PReluOnnx::opType() {
    return MNN::OpType_PReLU;
}
MNN::OpParameter PReluOnnx::type() {
    return MNN::OpParameter_PRelu;
}

void PReluOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                    std::vector<const onnx::TensorProto*> initializers) {
    auto preluPram = new MNN::PReluT;

    DCHECK(2 == onnxNode->input_size()) << "PRelu Input ERROR!";

    const onnx::TensorProto* slopeTp = initializers[0];
    DCHECK(slopeTp) << "PRelu Slope Input ERROR!";

    const float* slopeRawData = reinterpret_cast<const float*>(slopeTp->raw_data().data());
    DCHECK(slopeRawData) << "PRelu Slope Input ERROR!";

    const int slopeSize = slopeTp->raw_data().size() / sizeof(float);
    std::vector<float> slope(slopeSize);
    memcpy(slope.data(), slopeRawData, slopeSize * sizeof(float));

    preluPram->slopeCount = slopeSize;
    preluPram->slope      = slope;
    dstOp->main.value     = preluPram;
}

REGISTER_CONVERTER(PReluOnnx, PRelu);
