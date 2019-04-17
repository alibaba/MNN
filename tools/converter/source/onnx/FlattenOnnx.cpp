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
    return MNN::OpType_Reshape;
}

MNN::OpParameter FlattenOnnx::type() {
    return MNN::OpParameter_Reshape;
}

void FlattenOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                      std::vector<const onnx::TensorProto *> initializers) {
    auto param = new MNN::ReshapeT;

    int axis = 1;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto &attributeProto = onnxNode->attribute(i);
        const auto &attributeName  = attributeProto.name();
        if (attributeName == "axis") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT) << "Node Attribute ERROR";
            axis = attributeProto.i();
        }
    }

    param->dims.resize(axis + 1);
    for (int i = 0; i < axis; ++i) {
        param->dims[i] = 0;
    }
    param->dims[axis] = -1;

    dstOp->main.value = param;
}

REGISTER_CONVERTER(FlattenOnnx, Flatten);
