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

void ConvolutionTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, torchContext* context) {
    auto param = new MNN::Convolution2DT;
    const auto& inputs = node->inputs();
    const auto weight = inputs[1];
    const auto bias = inputs[2];
    const auto stride = getValue<std::vector<int64_t>>(inputs[3]);
    const auto padding = getValue<std::vector<int64_t>>(inputs[4]);
    const auto dialation = getValue<std::vector<int64_t>>(inputs[5]);
    const auto group = getValue<int64_t>(inputs[6]);
    std::vector<int> shape;
    param->common.reset(new MNN::Convolution2DCommonT);
    auto common = param->common.get();
    param->weight = getValue<float>(weight, shape);
    // weight format : NCHW
    common->outputCount = shape[0];
    common->inputCount = shape[1] * group;
    common->kernelY = shape[2];
    common->kernelX = shape[3];
    param->bias = getValue<float>(bias, shape);
    if (param->bias.empty()) {
        param->bias = std::vector<float>(common->outputCount, 0.f);
    }
    common->strideX = stride[0];
    common->strideY = stride[1];
    common->padX = padding[0];
    common->padY = padding[1];
    common->dilateX = dialation[0];
    common->dilateY = dialation[1];
    common->group   = static_cast<int32_t>(group);
    dstOp->main.value = param;
}

REGISTER_CONVERTER(ConvolutionTorch, conv2d);
