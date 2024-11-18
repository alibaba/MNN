//
//  UniqueOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2024/10/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(UniqueOnnx);

MNN::OpType UniqueOnnx::opType() {
    return MNN::OpType_Unique;
}

MNN::OpParameter UniqueOnnx::type() {
    return MNN::OpParameter_NONE;
}

void UniqueOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope* scope) {
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();
        if (attributeName == "axis") {
            MNN_ERROR("Don't support onnx Unique with axis\n");
        }
    }
    return;
}

REGISTER_CONVERTER(UniqueOnnx, Unique);
