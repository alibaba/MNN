//
//  SliceOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/7/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(SliceOnnx);

MNN::OpType SliceOnnx::opType() {
    return MNN::OpType_Slice;
}

MNN::OpParameter SliceOnnx::type() {
    return MNN::OpParameter_Slice;
}

void SliceOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                    std::vector<const onnx::TensorProto*> initializers) {
    auto param = new MNN::SliceT;
    int axis   = 1;

    std::vector<int> slicePoints(1);
    const auto attrSize = onnxNode->attribute_size();
    for (int i = 0; i < attrSize; ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();
        if (attributeName == "axes") {
            axis = attributeProto.ints(0);
        } else if (attributeName == "starts") {
            continue;
        } else if (attributeName == "ends") {
            slicePoints[0] = attributeProto.ints(0);
        }
    }

    param->axis        = axis;
    param->slicePoints = slicePoints;
    dstOp->main.value  = param;
}

REGISTER_CONVERTER(SliceOnnx, Slice);
