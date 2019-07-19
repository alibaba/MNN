//
//  UpsampleOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/05/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(UpsampleOnnx);

MNN::OpType UpsampleOnnx::opType() {
    return MNN::OpType_Interp;
}
MNN::OpParameter UpsampleOnnx::type() {
    return MNN::OpParameter_Interp;
}

void UpsampleOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                       std::vector<const onnx::TensorProto*> initializers) {
    auto interpParam = new MNN::InterpT;

    std::string mode;
    std::vector<float> scales;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();
        if (attributeName == "mode") {
            mode = attributeProto.s();
        } else if ((onnxNode->input_size() == 1) && (attributeName == "scales")) {
            scales.resize(attributeProto.floats_size());
            for (int j = 0; j < attributeProto.floats_size(); ++j) {
                scales[j] = attributeProto.floats(j);
            }
        }
    }

    if (onnxNode->input_size() != 1) {
        const onnx::TensorProto* scalesTp = initializers[0];
        if (!scalesTp) {
            DLOG(FATAL) << "Scales No TensorProto data!!!==> " << dstOp->name;
        }

        const float* raw_data = (const float*)scalesTp->raw_data().data();
        if (!raw_data) {
            DLOG(FATAL) << "Scales No raw data!!!==> " << dstOp->name;
        }

        int float_data_size = scalesTp->raw_data().size() / sizeof(float);
        for (int j = 0; j < float_data_size; ++j) {
            scales.push_back(raw_data[j]);
        }
    }

    // TODO defalut
    interpParam->widthScale  = 1.0f;
    interpParam->heightScale = 1.0f;
    if (scales.size() == 2) {
        interpParam->widthScale = scales[1];
    } else if (scales.size() == 3) {
        interpParam->widthScale  = scales[2];
        interpParam->heightScale = scales[1];
    } else if (scales.size() == 4) {
        interpParam->widthScale  = scales[3];
        interpParam->heightScale = scales[2];

        if (scales[1] != 1.0f) {
            DLOG(ERROR) << "Unsupported Upsample scales! ==> " << scales[1];
        }
    } else {
        DLOG(ERROR) << "Unsupported Upsample scales size! ==> " << scales.size();
    }

    // 1:near 2: bilinear 3: cubic
    if (mode == "nearest") {
        interpParam->resizeType = 1;
    } else if (mode == "bilinear" || mode == "linear") {
        interpParam->resizeType = 2;
    } else {
        DLOG(ERROR) << "Unsupported Upsample mode! ==> " << mode;
    }

    dstOp->main.value = interpParam;
}

REGISTER_CONVERTER(UpsampleOnnx, Upsample);
