//
//  Im2ColTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/11/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(Im2ColTorch);

MNN::OpType Im2ColTorch::opType() {
    return MNN::OpType_Im2Col;
}

MNN::OpParameter Im2ColTorch::type() {
    return MNN::OpParameter_Convolution2D;
}

std::vector<int> Im2ColTorch::inputTensorIdx() {
    return {0};
}

void Im2ColTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::Convolution2DT;
    param->common.reset(new MNN::Convolution2DCommonT);
    auto common = param->common.get();
    auto kernel_size = getValue<std::vector<int64_t>>(node->input(1));
    auto dilation = getValue<std::vector<int64_t>>(node->input(2));
    auto padding = getValue<std::vector<int64_t>>(node->input(3));
    auto stride = getValue<std::vector<int64_t>>(node->input(4));
    common->kernelY = kernel_size[0];
    common->kernelX = kernel_size[1];
    common->strideY = stride[0];
    common->strideX = stride[1];
    common->padY = padding[0];
    common->padX = padding[1];
    common->dilateY = dilation[0];
    common->dilateX = dilation[1];
    dstOp->main.value = param;
}
REGISTER_CONVERTER(Im2ColTorch, im2col);

DECLARE_OP_CONVERTER(Col2ImTorch);

MNN::OpType Col2ImTorch::opType() {
    return MNN::OpType_Col2Im;
}

MNN::OpParameter Col2ImTorch::type() {
    return MNN::OpParameter_Convolution2D;
}

std::vector<int> Col2ImTorch::inputTensorIdx() {
    return {0, 1};
}

void Col2ImTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::Convolution2DT;
    param->common.reset(new MNN::Convolution2DCommonT);
    auto common = param->common.get();
    auto kernel_size = getValue<std::vector<int64_t>>(node->input(2));
    auto dilation = getValue<std::vector<int64_t>>(node->input(3));
    auto padding = getValue<std::vector<int64_t>>(node->input(4));
    auto stride = getValue<std::vector<int64_t>>(node->input(5));
    common->kernelY = kernel_size[0];
    common->kernelX = kernel_size[1];
    common->strideY = stride[0];
    common->strideX = stride[1];
    common->padY = padding[0];
    common->padX = padding[1];
    common->dilateY = dilation[0];
    common->dilateX = dilation[1];
    dstOp->main.value = param;
}
REGISTER_CONVERTER(Col2ImTorch, col2im);
