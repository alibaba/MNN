//
//  CumPordTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2022/05/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(CumPordTorch);

MNN::OpType CumPordTorch::opType() {
    return MNN::OpType_CumProd;
}
MNN::OpParameter CumPordTorch::type() {
    return MNN::OpParameter_Axis;
}
std::vector<int> CumPordTorch::inputTensorIdx() {
    return {0};
}

void CumPordTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::AxisT;
    param->axis = getValue<int64_t>(node->input(1));
    dstOp->main.value = param;
}

REGISTER_CONVERTER(CumPordTorch, cumprod);
