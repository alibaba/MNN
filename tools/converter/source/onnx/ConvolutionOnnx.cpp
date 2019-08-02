//
//  ConvolutionOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ConvolutionOnnx);

MNN::OpType ConvolutionOnnx::opType() {
    return MNN::OpType_Convolution;
}

MNN::OpParameter ConvolutionOnnx::type() {
    return MNN::OpParameter_Convolution2D;
}

void ConvolutionOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                          std::vector<const onnx::TensorProto*> initializers) {
    auto convParam = new MNN::Convolution2DT;

    const int size = initializers.size();
    DCHECK(size <= 2 && size >= 1) << "Convolution Input ERROR!";

    const auto weightProto = initializers[0];
    const auto biasProto   = size == 2 ? initializers[1] : nullptr;

    const auto weightDimSize = weightProto->dims_size();
    DCHECK(weightDimSize == 4) << "Convolution Weight ERROR! ==> " << weightProto->name();

    int co = weightProto->dims(0);
    int ci = weightProto->dims(1); // depthwise convolution, this value equal to 1
    int kh = weightProto->dims(2);
    int kw = weightProto->dims(3);

    int group      = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    int stride_h   = 1;
    int stride_w   = 1;
    int padX       = 0;
    int padY       = 0;

    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();

        if (attributeName == "dilations") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS) << "Node Attribute ERROR";
            DCHECK(attributeProto.ints_size() == 2) << "Node Attribute ERROR";
            dilation_h = attributeProto.ints(0);
            dilation_w = attributeProto.ints(1);
        } else if (attributeName == "group") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT) << "Node Attribute ERROR";
            group = attributeProto.i();
        } else if (attributeName == "strides") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS) << "Node Attribute ERROR";
            DCHECK(attributeProto.ints_size() == 2) << "Node Attribute ERROR";
            stride_h = attributeProto.ints(0);
            stride_w = attributeProto.ints(1);
        } else if (attributeName == "auto_pad") {
            DCHECK(attributeProto.strings(0) == "NOTSET") << "auto_pad now only support NOTSET";
        } else if (attributeName == "pads") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS) << "Node Attribute ERROR";
            DCHECK(attributeProto.ints_size() == 4) << "Node Attribute ERROR";
            padX = attributeProto.ints(1);
            padY = attributeProto.ints(0);
            int padX_end = attributeProto.ints(3);
            int padY_end = attributeProto.ints(2);
            DCHECK((padX == padX_end) && (padY == padY_end)) << "Asymmetrical pads in convolution is not supported";
        }
    }

    const int weightSize = co * ci * kh * kw;
    convParam->weight.resize(weightSize);

    if (weightProto->float_data_size() != 0) {
        for (int i = 0; i < weightSize; ++i) {
            convParam->weight[i] = weightProto->float_data(i);
        }
    } else if (weightProto->raw_data().data()) {
        ::memcpy(convParam->weight.data(), weightProto->raw_data().data(), weightSize * sizeof(float));
    }

    convParam->bias.resize(co);
    if (biasProto) {
        if (biasProto->float_data_size() != 0) {
            for (int i = 0; i < co; ++i) {
                convParam->bias[i] = biasProto->float_data(i);
            }
        } else if (biasProto->raw_data().data()) {
            ::memcpy(convParam->bias.data(), biasProto->raw_data().data(), co * sizeof(float));
        }
    } else {
        for (int i = 0; i < co; ++i) {
            convParam->bias[i] = .0f;
        }
    }

    convParam->common   = std::unique_ptr<MNN::Convolution2DCommonT>(new MNN::Convolution2DCommonT);
    auto& common        = convParam->common;
    common->relu        = false;
    common->group       = group;
    common->outputCount = co;
    common->inputCount  = group == 1 ? ci : group; // conv set inputCount to be ci, dw to be group
    common->kernelX     = kw;
    common->kernelY     = kh;
    common->dilateX     = dilation_w;
    common->dilateY     = dilation_h;
    common->strideX     = stride_w;
    common->strideY     = stride_h;
    common->padX        = padX;
    common->padY        = padY;
    common->padMode     = MNN::PadMode_CAFFE;

    dstOp->main.value = convParam;
}

REGISTER_CONVERTER(ConvolutionOnnx, Conv);
