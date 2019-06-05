//
//  GatherOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(GatherOnnx);

MNN::OpType GatherOnnx::opType() {
    return MNN::OpType_Gather;
}
MNN::OpParameter GatherOnnx::type() {
    return MNN::OpParameter_Gather;
}

void GatherOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                     std::vector<const onnx::TensorProto*> initializers) {
    auto para  = new MNN::GatherT;
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

REGISTER_CONVERTER(GatherOnnx, Gather);
