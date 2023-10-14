//
//  SplitOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(SplitOnnx);

MNN::OpType SplitOnnx::opType() {
    return MNN::OpType_Slice;
}

MNN::OpParameter SplitOnnx::type() {
    return MNN::OpParameter_Slice;
}

void SplitOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                    OnnxScope* scope) {
    auto param = new MNN::SliceT;
    // Default axis = 0: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split
    int axis   = 0;
    std::vector<int> slicePoints;
    const auto attrSize = onnxNode->attribute_size();
    for (int i = 0; i < attrSize; ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();
        if (attributeName == "axis") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT) << "Node Attribute ERROR";
            axis = attributeProto.i();
        } else if (attributeName == "split") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS) << "Node Attribute ERROR";
            const int splitSize = attributeProto.ints_size();
            for (int k = 0; k < splitSize; ++k) {
                slicePoints.push_back(attributeProto.ints(k));
            }
        }
    }
    param->axis        = axis;
    param->slicePoints = slicePoints;
    param->sourceType = MNN::NetSource_TORCH;
    dstOp->main.value  = param;
}

REGISTER_CONVERTER(SplitOnnx, Split);
