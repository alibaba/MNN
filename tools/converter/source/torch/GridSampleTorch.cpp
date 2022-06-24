//
//  GridSampleTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2022/04/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(GridSampleTorch);

MNN::OpType GridSampleTorch::opType() {
    return MNN::OpType_GridSample;
}
MNN::OpParameter GridSampleTorch::type() {
    return MNN::OpParameter_GridSample;
}
std::vector<int> GridSampleTorch::inputTensorIdx() {
    return {0, 1};
}

void GridSampleTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto gridSampleParam = new MNN::GridSampleT;
    int mode = getValue<int64_t>(node->input(2));
    if (mode == 0 || mode == 1) {
        gridSampleParam->mode = static_cast<MNN::SampleMode>(mode);
    } else {
        LOG(FATAL) << "Unknown mode for " << dstOp->name << "!";
    }
    int padding_mode = getValue<int64_t>(node->input(3));
    if (padding_mode == 0 || padding_mode == 1 || padding_mode == 2) {
        gridSampleParam->paddingMode = static_cast<MNN::BorderMode>(mode);
    } else {
        LOG(FATAL) << "Unknown padding for " << dstOp->name << "!";
    }
    gridSampleParam->alignCorners = getValue<bool>(node->input(4));
    dstOp->main.value = gridSampleParam;
}

REGISTER_CONVERTER(GridSampleTorch, grid_sampler);
