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

MNN::OpParameter ROIAlignOnnx::type() { return MNN::OpParameter_RoiParameters; }

void ROIAlignOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode, OnnxScope* scope) {
    auto roiAlign = new MNN::RoiParametersT;
    // default value from onnx docs
    roiAlign->pooledHeight = roiAlign->pooledWidth = 1;
    roiAlign->poolType = MNN::PoolType_AVEPOOL;
    roiAlign->spatialScale = 1;
    roiAlign->samplingRatio = 0;
    roiAlign->aligned = true;
    
    const auto attrSize = onnxNode->attribute_size();
    for (int i = 0; i < attrSize; ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();

        if (attributeName == "output_height") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT) << "Node Attribute ERROR";
            roiAlign->pooledHeight = attributeProto.i();
        } else if (attributeName == "output_width") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT) << "Node Attribute ERROR";
            roiAlign->pooledWidth  = attributeProto.i();
        } else if (attributeName == "mode") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_STRING) << "Node Attribute ERROR";
            roiAlign->poolType = (attributeProto.s() == "max" ? MNN::PoolType_MAXPOOL : MNN::PoolType_AVEPOOL);
        } else if (attributeName == "spatial_scale") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_FLOAT) << "Node Attribute ERROR";
            roiAlign->spatialScale = attributeProto.f();
        } else if (attributeName == "sampling_ratio") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT) << "Node Attribute ERROR";
            roiAlign->samplingRatio = attributeProto.i();
        } else if (attributeName == "coordinate_transformation_mode") {
            roiAlign->aligned = (attributeProto.s() == "half_pixel"); // opset_version = 16
        } else {
            DLOG(ERROR) << "TODO!";
        }
    }

    dstOp->main.value  = roiAlign;
};

REGISTER_CONVERTER(ROIAlignOnnx, RoiAlign);
