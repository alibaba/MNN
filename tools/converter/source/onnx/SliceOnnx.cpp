//
//  SliceOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/07/16.
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
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS) << "Node Attribute ERROR";
            axis = attributeProto.ints(0);
        } else if (attributeName == "starts") {
            continue;
        } else if (attributeName == "ends") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS) << "Node Attribute ERROR";
            slicePoints[0] = attributeProto.ints(0);
        }
    }
    DCHECK(1 == axis) << "Only support axis equal to 1";
    param->axis        = axis;
    param->slicePoints = slicePoints;
    dstOp->main.value  = param;
}

// REGISTER_CONVERTER(SliceOnnx, Slice);

DECLARE_OP_CONVERTER(SplitOnnx);

MNN::OpType SplitOnnx::opType() {
    return MNN::OpType_Slice;
}

MNN::OpParameter SplitOnnx::type() {
    return MNN::OpParameter_Slice;
}

void SplitOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                    std::vector<const onnx::TensorProto*> initializers) {
    auto param = new MNN::SliceT;
    int axis   = 1;
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
                if (k == 0) {
                    slicePoints.push_back(attributeProto.ints(k));
                } else {
                    slicePoints.push_back(attributeProto.ints(k) + slicePoints.back());
                }
            }
        }
    }
    DCHECK(1 == axis) << "Only support axis equal to 1";
    param->axis        = axis;
    param->slicePoints = slicePoints;
    dstOp->main.value  = param;
}

REGISTER_CONVERTER(SplitOnnx, Split);
