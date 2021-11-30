//
//  UpsampleTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(UpsampleTorch);

MNN::OpType UpsampleTorch::opType() {
    return MNN::OpType_Interp;
}
MNN::OpParameter UpsampleTorch::type() {
    return MNN::OpParameter_Interp;
}
std::vector<int> UpsampleTorch::inputTensorIdx() {
    return {0};
}

void UpsampleTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::InterpT;
    std::string opType =  getRealOpType(node);
    if (opType == "upsample_nearest2d") {
        param->resizeType = 1;
        if (node->inputs().size() == 3) {
            auto scales = getValue<std::vector<double>>(node->input(2));
            param->heightScale = scales[0];
            param->widthScale = scales[1];
        } else if (node->inputs().size() == 4) {
            param->heightScale = getValue<float>(node->input(2));
            param->widthScale = getValue<float>(node->input(3));
        }
    } else if (opType == "upsample_bilinear2d") {
        param->resizeType = 2;
        if (toIValue(node->input(1))) {
            auto output_size = getValue<std::vector<int64_t>>(node->input(1));
            if (output_size.size() == 2) {
                param->outputWidth = output_size[0];
                param->outputHeight = output_size[1];
            }
        } else {
            const auto inputName = node->input(1)->debugName();
            scope->addInputForOp(dstOp, inputName, true);
        }
        param->alignCorners = getValue<bool>(node->input(2));
        if (node->inputs().size() == 4) {
            auto scales = getValue<std::vector<double>>(node->input(3));
            if (scales.size() == 2) {
                param->heightScale = scales[0];
                param->widthScale = scales[1];
            }
            else { param->heightScale = 2; param->widthScale = 2; }
        } else if (node->inputs().size() == 5) {
            param->heightScale = getValue<float>(node->input(3));
            param->widthScale = getValue<float>(node->input(4));
        }
    } else if (opType == "upsample_bicubic2d") {
        param->resizeType = 3;
        param->alignCorners = getValue<bool>(node->input(2));
        auto scales = getValue<std::vector<float>>(node->input(2));
        param->heightScale = scales[0];
        param->widthScale = scales[1];
    }
    dstOp->main.value = param;
}

// aten::upsample_bilinear2d(Tensor self, int[] output_size, bool align_corners, float? scales_h, float? scales_w) -> Tensor
// aten::upsample_bilinear2d(Tensor self, int[] output_size, bool align_corners, float[]? scale_factors) -> Tensor
REGISTER_CONVERTER(UpsampleTorch, upsample_bilinear2d);
// aten::upsample_nearest2d(Tensor self, int[] output_size, float? scales_h, float? scales_w) -> Tensor
// aten::upsample_nearest2d(Tensor self, int[] output_size, float[]? scale_factors) -> Tensor
REGISTER_CONVERTER(UpsampleTorch, upsample_nearest2d);
// aten::upsample_bicubic2d(Tensor self, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
REGISTER_CONVERTER(UpsampleTorch, upsample_bicubic2d);
