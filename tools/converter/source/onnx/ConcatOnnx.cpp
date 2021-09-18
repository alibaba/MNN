//
//  ConcatOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ConcatOnnx);

MNN::OpType ConcatOnnx::opType() {
    return MNN::OpType_Concat;
}
MNN::OpParameter ConcatOnnx::type() {
    return MNN::OpParameter_Axis;
}

void ConcatOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode, OnnxScope* scope) {
    auto para  = new MNN::AxisT;
    para->axis = 0;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();
        if (attributeName == "axis") {
            para->axis = attributeProto.i();
        }
    }

    dstOp->main.value = para;
}

REGISTER_CONVERTER(ConcatOnnx, Concat);
