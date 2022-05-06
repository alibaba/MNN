//
//  PoolTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(PoolTorch);

MNN::OpType PoolTorch::opType() {
    return MNN::OpType_Pooling;
}
MNN::OpParameter PoolTorch::type() {
    return MNN::OpParameter_Pool;
}
std::vector<int> PoolTorch::inputTensorIdx() {
    return {0};
}

void PoolTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::PoolT;
    std::string opType = getRealOpType(node);
    const auto& inputs = node->inputs();
    if (opType.find("adaptive") == std::string::npos) {
        const auto kernel_size = getValue<std::vector<int64_t>>(inputs[1]);
        param->kernelY = kernel_size[0];
        param->kernelX = kernel_size[1];
        if (inputs.size() > 2) {
            const auto stride = getValue<std::vector<int64_t>>(inputs[2]);
            if (stride.size() == 2) {
                param->strideY = stride[0];
                param->strideX = stride[1];
            } else {
                param->strideX = 2;
                param->strideY = 2;
            }
        }
        if (inputs.size() > 3) {
            const auto padding = getValue<std::vector<int64_t>>(inputs[3]);
            param->padY = padding[0];
            param->padX = padding[1];
        }
        if (inputs.size() > 5) {
            // const auto dialation = getValue<std::vector<int64_t>>(inputs[4]);
            const auto ceil_mode = getValue<bool>(inputs[5]);
            param->ceilModel = ceil_mode;
        }
    } else {
        const auto outputSize = getValue<std::vector<int64_t>>(inputs[1]);
        if (outputSize[0] == 1 && outputSize[1] == 1) {
            param->isGlobal = true;
        } else {
            // TODO: support adaptive pooling
            param->kernelX = 1;
            param->kernelY = 1;
            param->strideX = 1;
            param->strideY = 1;
            param->padX = 0;
            param->padY = 0;
            param->ceilModel = false;
        }
    }
    param->type = opType.find("max") == std::string::npos ? MNN::PoolType_AVEPOOL : MNN::PoolType_MAXPOOL;
    dstOp->main.value = param;
}

REGISTER_CONVERTER(PoolTorch, max_pool2d);
REGISTER_CONVERTER(PoolTorch, avg_pool2d);
REGISTER_CONVERTER(PoolTorch, adaptive_avg_pool2d);
