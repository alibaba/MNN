//
//  ROIAlignOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2021/11/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ROIAlignOnnx);

MNN::OpType ROIAlignOnnx::opType() { return MNN::OpType_ROIAlign; }

MNN::OpParameter ROIAlignOnnx::type() { return MNN::OpParameter_RoiAlign; }

void ROIAlignOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode, OnnxScope* scope) {
    auto roiAlign = new MNN::RoiAlignT;

    const auto attrSize = onnxNode->attribute_size();
    for (int i = 0; i < attrSize; ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();

        if (attributeName == "output_size") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS) << "Node Attribute ERROR";
            DCHECK(attributeProto.ints_size() == 2) << "Node Attribute ERROR";
            roiAlign->pooledHeight = attributeProto.ints(0);
            roiAlign->pooledWidth  = attributeProto.ints(1);
        } else if (attributeName == "spatial_scale") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_FLOAT) << "Node Attribute ERROR";
            roiAlign->spatialScale = attributeProto.f();
        } else if (attributeName == "sampling_ratio") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT) << "Node Attribute ERROR";
            roiAlign->samplingRatio = attributeProto.i();
        } else if (attributeName == "aligned") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT) << "Node Attribute ERROR";
            roiAlign->aligned = attributeProto.i() == 0 ? false : true;
        } else {
            DLOG(ERROR) << "TODO!";
        }
    }

    roiAlign->poolMode = MNN::PoolMode::PoolMode_AvePool;
    dstOp->main.value  = roiAlign;
};

REGISTER_CONVERTER(ROIAlignOnnx, ROIAlign);