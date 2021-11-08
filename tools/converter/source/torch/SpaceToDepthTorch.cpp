//
//  SpaceToDepthTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/06/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(DepthToSpaceTorch);

MNN::OpType DepthToSpaceTorch::opType() {
    return MNN::OpType_DepthToSpace;
}

MNN::OpParameter DepthToSpaceTorch::type() {
    return MNN::OpParameter_DepthSpaceParam;
}

std::vector<int> DepthToSpaceTorch::inputTensorIdx() {
    return {0};
}

void DepthToSpaceTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::DepthSpaceParamT;
    std::string opType = getRealOpType(node);
    const auto upscale = node->inputs()[1];
    param->blockSize = getValue<int64_t>(upscale);
    dstOp->main.value = param;
}

REGISTER_CONVERTER(DepthToSpaceTorch, pixel_shuffle);
