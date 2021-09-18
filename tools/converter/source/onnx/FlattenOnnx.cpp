//
//  FlattenOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(FlattenOnnx);

MNN::OpType FlattenOnnx::opType() {
    return MNN::OpType_Flatten;
}

MNN::OpParameter FlattenOnnx::type() {
    return MNN::OpParameter_Flatten;
}

void FlattenOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                      OnnxScope* scope) {
    auto param = new MNN::FlattenT;

    // Ref https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten, Default is 1
    int axis = 1;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto &attributeProto = onnxNode->attribute(i);
        const auto &attributeName  = attributeProto.name();
        if (attributeName == "axis") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT) << "Node Attribute ERROR";
            axis = attributeProto.i();
        }
    }
    param->axis = axis;
    dstOp->main.value = param;
}

REGISTER_CONVERTER(FlattenOnnx, Flatten);
