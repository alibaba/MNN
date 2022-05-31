//
//  GridSampleOnnxClassic.cpp
//  MNNConverter
//
//  Created by MNN on 2022/05/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(GridSampleOnnxClassic);

MNN::OpType GridSampleOnnxClassic::opType(){
    return MNN::OpType_GridSample;
}

MNN::OpParameter GridSampleOnnxClassic::type(){
    return MNN::OpParameter_GridSample;
}

void GridSampleOnnxClassic::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope* scope){
    
    auto gridSampleParam = new MNN::GridSampleT;

    gridSampleParam->mode = MNN::SampleMode_BILINEAR;
    gridSampleParam->paddingMode = MNN::BorderMode_ZEROS;
    gridSampleParam->alignCorners = false;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto &attributeProto = onnxNode->attribute(i);
        const auto &attributeName  = attributeProto.name();
        if (attributeName == "mode") {
            gridSampleParam->mode = MNN::SampleMode_BILINEAR;
            if (attributeProto.s() == "bilinear") {
                gridSampleParam->mode = MNN::SampleMode_BILINEAR;
            } else if (attributeProto.s() == "nearest") {
                gridSampleParam->mode = MNN::SampleMode_NEAREST;
            } else {
                LOG_INFO.stream() << "Don't support mode " << attributeProto.s();
            }
        }
        if (attributeName == "padding_mode") {
            gridSampleParam->paddingMode = MNN::BorderMode_ZEROS;
            if (attributeProto.s() == "zeros") {
                gridSampleParam->paddingMode = MNN::BorderMode_ZEROS;
            } else if (attributeProto.s() == "border") {
                gridSampleParam->paddingMode = MNN::BorderMode_CLAMP;
            } else if (attributeProto.s() == "reflection") {
                gridSampleParam->paddingMode = MNN::BorderMode_REFLECTION;
            } else {
                LOG_INFO.stream() << "Don't support padding_mode " << attributeProto.s();
            }
        }
        if (attributeName == "align_corners") {
            gridSampleParam->alignCorners = attributeProto.i();
        }
    }
    
    dstOp->main.value = gridSampleParam;
}

REGISTER_CONVERTER(GridSampleOnnxClassic, GridSample);
