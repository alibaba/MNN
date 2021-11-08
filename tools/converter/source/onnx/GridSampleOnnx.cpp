//
//  GridSampleOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(GridSampleOnnx);

MNN::OpType GridSampleOnnx::opType(){
    return MNN::OpType_GridSample;
}

MNN::OpParameter GridSampleOnnx::type(){
    return MNN::OpParameter_GridSample;
}

void GridSampleOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope* scope){
    
    auto gridSampleParam = new MNN::GridSampleT;

    gridSampleParam->mode = MNN::SampleMode_BILINEAR;
    gridSampleParam->paddingMode = MNN::BorderMode_ZEROS;
    gridSampleParam->alignCorners = false;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto &attributeProto = onnxNode->attribute(i);
        const auto &attributeName  = attributeProto.name();
        if (attributeName == "mode") {
            switch (attributeProto.i()) {
                case 0:
                    gridSampleParam->mode = MNN::SampleMode_BILINEAR;
                    break;
                case 1:
                    gridSampleParam->mode = MNN::SampleMode_NEAREST;
                    break;
                default:
                    LOG(FATAL) << "Unknown mode for " << onnxNode->name() << "!";
                    break;
            }
        }
        if (attributeName == "padding_mode") {
            switch (attributeProto.i()) {
                case 0:
                    gridSampleParam->paddingMode = MNN::BorderMode_ZEROS;
                    break;
                case 1:
                    gridSampleParam->paddingMode = MNN::BorderMode_CLAMP;
                    break;
                case 2:
                    gridSampleParam->paddingMode = MNN::BorderMode_REFLECTION;
                    break;
                default:
                    LOG(FATAL) << "Unknown padding mode for " << onnxNode->name() << "!";
                    break;
            }
        }
        if (attributeName == "align_corners") {
            gridSampleParam->alignCorners = attributeProto.i();
        }
    }
    
    dstOp->main.value = gridSampleParam;
}

// REGISTER_CONVERTER(GridSampleOnnx, GridSample);

// When we export torch.nn.functional.grid_sample to onnx, it's called GridSampler rather than GridSample,
// thus, we have to add the "r"
#define REGISTER_CONVERTER_r(name, opType) static onnxOpConverterRegister<name> _Convert_##opType(#opType"r")
REGISTER_CONVERTER_r(GridSampleOnnx, GridSample);
