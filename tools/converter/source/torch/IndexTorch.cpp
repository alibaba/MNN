//
//  IndexTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2022/04/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(IndexTorch);

MNN::OpType IndexTorch::opType() {
    return MNN::OpType_Extra;
}
MNN::OpParameter IndexTorch::type() {
    return MNN::OpParameter_Extra;
}
std::vector<int> IndexTorch::inputTensorIdx() {
    return {-1};
}

void IndexTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto extra = new MNN::ExtraT;
    dstOp->main.value = extra;
    extra->engine     = "Torch";
    extra->type       = getRealOpType(node);
}

REGISTER_CONVERTER(IndexTorch, index);
REGISTER_CONVERTER(IndexTorch, index_stridedslice);
REGISTER_CONVERTER(IndexTorch, index_put);
