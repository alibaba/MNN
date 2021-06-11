//
//  SliceTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(SliceTorch);

MNN::OpType SliceTorch::opType() {
    return MNN::OpType_Slice;
}
MNN::OpParameter SliceTorch::type() {
    return MNN::OpParameter_Slice;
}
std::vector<int> SliceTorch::inputTensorIdx() {
    return {0};
}

void SliceTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, torchContext* context) {
    auto param = new MNN::SliceT;
    const std::string opType = node->kind().toUnqualString();
    if (opType == "chunk") {
        param->axis = getValue<int64_t>(node->input(2));
        param->sourceType = MNN::NetSource_TENSORFLOW;
    } else if (opType == "ListUnpack") {
        param->axis = 0;
        param->sourceType = MNN::NetSource_TENSORFLOW;
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(SliceTorch, chunk);
REGISTER_CONVERTER(SliceTorch, ListUnpack);

DECLARE_OP_CONVERTER(StridedSliceTorch);

MNN::OpType StridedSliceTorch::opType() {
    return MNN::OpType_StridedSlice;
}
MNN::OpParameter StridedSliceTorch::type() {
    return MNN::OpParameter_StridedSliceParam;
}
std::vector<int> StridedSliceTorch::inputTensorIdx() {
    return {0};
}

void StridedSliceTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, torchContext* context) {
    auto param = new MNN::StridedSliceParamT;
    const auto inputs = node->inputs();
    int dim   = getValue<int64_t>(inputs[1]);
    int start = getValue<int64_t>(inputs[2]);
    int end   = getValue<int64_t>(inputs[3]);
    int step  = getValue<int64_t>(inputs[4]);
    dstOp->main.value = param;
}

REGISTER_CONVERTER(StridedSliceTorch, slice);
