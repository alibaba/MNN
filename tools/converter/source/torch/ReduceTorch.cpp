//
//  ReduceTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(ReduceTorch);

MNN::OpType ReduceTorch::opType() {
    return MNN::OpType_Reduction;
}

MNN::OpParameter ReduceTorch::type() {
    return MNN::OpParameter_ReductionParam;
}

std::vector<int> ReduceTorch::inputTensorIdx() {
    return {0};
}

void ReduceTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, torchContext* context) {
    const auto inputs = node->inputs();
    auto param = new MNN::ReductionParamT;
    std::string opType = node->kind().toUnqualString();
    if (opType == "mean") {
        param->operation = MNN::ReductionType_MEAN;
        const auto dims = getValue<std::vector<int64_t>>(inputs[1]);
        for (int i : dims) {
            param->dim.push_back(i);
        }
        param->keepDims = getValue<bool>(inputs[2]);
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(ReduceTorch, mean);
