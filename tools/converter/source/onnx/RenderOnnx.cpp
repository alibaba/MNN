//
//  RenderOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2023/06/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(TextureOnnx);

MNN::OpType TextureOnnx::opType() {
    return MNN::OpType_Texture;
}
MNN::OpParameter TextureOnnx::type() {
    return MNN::OpParameter_GridSample;
}
/**
 # Convert filter mode to internal enumeration.
 filter_mode_dict = {'nearest': 0, 'linear': 1, 'linear-mipmap-nearest': 2, 'linear-mipmap-linear': 3}
 filter_mode_enum = filter_mode_dict[filter_mode]

 # Convert boundary mode to internal enumeration.
 boundary_mode_dict = {'cube': 0, 'wrap': 1, 'clamp': 2, 'zero': 3}
 boundary_mode_enum = boundary_mode_dict[boundary_mode]
 */
void TextureOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                    OnnxScope* scope) {
    dstOp->main.value = new MNN::GridSampleT;
    auto grid = dstOp->main.AsGridSample();
    for (int i=0; i<onnxNode->attribute_size(); ++i) {
        auto attr = onnxNode->attribute(i);
        if (attr.name() == "filter_mode") {
            if (attr.i() == 1) {
                grid->mode = MNN::SampleMode_BILINEAR;
            } else if (attr.i() == 0) {
                grid->mode = MNN::SampleMode_NEAREST;
            }
            continue;
        }
        if (attr.name() == "boundary_mode") {
            if (3 == attr.i()) {
                grid->paddingMode = MNN::BorderMode_ZEROS;
            } else if (2 == attr.i()) {
                grid->paddingMode = MNN::BorderMode_CLAMP;
            } else if (1 == attr.i()) {
                grid->paddingMode = MNN::BorderMode_CLAMP;
            } else if (0 == attr.i()) {
                grid->paddingMode = MNN::BorderMode_CUBE;
            }
            continue;
        }
    }
    dstOp->main.AsGridSample();
    dstOp->defaultDimentionFormat = MNN::MNN_DATA_FORMAT_NCHW;
}

REGISTER_CONVERTER(TextureOnnx, texture);
REGISTER_CONVERTER(TextureOnnx, texture_mipmap);


DECLARE_OP_CONVERTER(RasterGradOnnx);

MNN::OpType RasterGradOnnx::opType() {
    return MNN::OpType_RasterDiff;
}
MNN::OpParameter RasterGradOnnx::type() {
    return MNN::OpParameter_NONE;
}

void RasterGradOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                    OnnxScope* scope) {
    dstOp->defaultDimentionFormat = MNN::MNN_DATA_FORMAT_NCHW;
}

REGISTER_CONVERTER(RasterGradOnnx, raster_grad);
