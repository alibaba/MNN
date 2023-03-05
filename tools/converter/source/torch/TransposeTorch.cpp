//
//  TransposeTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(PermuteTorch);

MNN::OpType PermuteTorch::opType() {
    return MNN::OpType_Permute;
}
MNN::OpParameter PermuteTorch::type() {
    return MNN::OpParameter_Permute;
}
std::vector<int> PermuteTorch::inputTensorIdx() {
    return {0};
}

void PermuteTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::PermuteT;
    auto type = getRealOpType(node);
    if (type == "numpy_T" || type == "t") {
        param->dims = {1, 0};
    } else {
        auto dims = getValue<std::vector<int64_t>>(node->input(1));
        param->dims.resize(dims.size());
        for (int i = 0; i < dims.size(); i++) {
            param->dims[i] = dims[i];
        }
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(PermuteTorch, permute);
REGISTER_CONVERTER(PermuteTorch, numpy_T);
REGISTER_CONVERTER(PermuteTorch, t);

DECLARE_OP_CONVERTER(TransposeTorch);

MNN::OpType TransposeTorch::opType() {
    return MNN::OpType_Extra;
}
MNN::OpParameter TransposeTorch::type() {
    return MNN::OpParameter_Extra;
}
std::vector<int> TransposeTorch::inputTensorIdx() {
    return {-1};
}

void TransposeTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto extra = new MNN::ExtraT;
    dstOp->main.value = extra;
    extra->engine     = "Torch";
    extra->type       = "transpose";
}

// aten::transpose(self : Tensor, dim0 : int , dim1 : int)
REGISTER_CONVERTER(TransposeTorch, transpose);
