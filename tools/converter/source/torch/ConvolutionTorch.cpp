//
//  ConvolutionTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(ConvolutionTorch);

MNN::OpType ConvolutionTorch::opType() {
    return MNN::OpType_Convolution;
}
MNN::OpParameter ConvolutionTorch::type() {
    return MNN::OpParameter_Convolution2D;
}
std::vector<int> ConvolutionTorch::inputTensorIdx() {
    return {0};
}

void ConvolutionTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::Convolution2DT;
    param->common.reset(new MNN::Convolution2DCommonT);
    auto common = param->common.get();
    // input, weight, bias, stride, padding, dialation
    const auto& inputs = node->inputs();
    const auto weight = inputs[1];
    const auto bias = inputs[2];
    const auto stride = getValue<std::vector<int64_t>>(inputs[3]);
    const auto padding = getValue<std::vector<int64_t>>(inputs[4]);
    const auto dialation = getValue<std::vector<int64_t>>(inputs[5]);
    std::vector<int> weightShape, biasShape;
    param->weight = getValue<float>(weight, weightShape);
    param->bias = getValue<float>(bias, biasShape);
    if (param->bias.empty()) {
        param->bias = std::vector<float>(weightShape[0], 0.f);
    }
    std::string opType = getRealOpType(node);
    if (opType == "conv2d") {
        common->group   = static_cast<int>(getValue<int64_t>(inputs[6]));
    } else if (opType == "convolution") {
        common->group   = static_cast<int>(getValue<int64_t>(inputs[8]));
    }
    bool conv1d = (stride.size() == 1 && weightShape.size() == 3);
    if (conv1d) {
        common->strideX = 1;
        common->strideY = stride[0];
        common->padX = 0;
        common->padY = padding[0];
        common->dilateX = 1;
        common->dilateY = dialation[0];
        // weight format : NCH
        common->outputCount = weightShape[0];
        common->inputCount = weightShape[1] * common->group;
        common->kernelY = weightShape[2];
        common->kernelX = 1;
    } else {
        common->strideY = stride[0];
        common->strideX = stride[1];
        common->padY = padding[0];
        common->padX = padding[1];
        common->dilateY = dialation[0];
        common->dilateX = dialation[1];
        // weight format : NCHW
        common->outputCount = weightShape[0];
        common->inputCount = weightShape[1] * common->group;
        common->kernelY = weightShape[2];
        common->kernelX = weightShape[3];
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(ConvolutionTorch, conv2d);
REGISTER_CONVERTER(ConvolutionTorch, convolution);
