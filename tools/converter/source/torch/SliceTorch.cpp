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

void SliceTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::SliceT;
    const std::string opType = getRealOpType(node);
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
    return MNN::OpType_Extra;
}
MNN::OpParameter StridedSliceTorch::type() {
    return MNN::OpParameter_Extra;
}
std::vector<int> StridedSliceTorch::inputTensorIdx() {
    return {-1};
}

void StridedSliceTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto extra        = new MNN::ExtraT;
    dstOp->main.value = extra;
    extra->engine     = "Torch";
    extra->type       = getRealOpType(node);
    if (node->inputs().size() > 1) {
        auto dim = node->input(1);
        if (toIValue(dim)) {
            std::unique_ptr<MNN::AttributeT> dimAttr(new MNN::AttributeT);
            dimAttr->key = "dim";
            dimAttr->i = getValue<int64_t>(dim);
            extra->attr.emplace_back(std::move(dimAttr));
        }
    }
    if (node->inputs().size() > 2) {
        auto start = node->input(2);
        if (toIValue(start)) {
            std::unique_ptr<MNN::AttributeT> startAttr(new MNN::AttributeT);
            startAttr->key = "start";
            startAttr->i = getValue<int64_t>(start);
            extra->attr.emplace_back(std::move(startAttr));
        }
    }
    if (node->inputs().size() > 3) {
        auto end = node->input(3);
        if (toIValue(end)) {
            std::unique_ptr<MNN::AttributeT> endAttr(new MNN::AttributeT);
            endAttr->key = "end";
            endAttr->i = std::min(getValue<int64_t>(end), static_cast<int64_t>(std::numeric_limits<int>::max()));
            extra->attr.emplace_back(std::move(endAttr));
        }
    }
    if (node->inputs().size() > 4) {
        auto stride = node->input(4);
        if (toIValue(stride)) {
            std::unique_ptr<MNN::AttributeT> strideAttr(new MNN::AttributeT);
            strideAttr->key = "stride";
            strideAttr->i = getValue<int64_t>(stride);
            extra->attr.emplace_back(std::move(strideAttr));
        }
    } else {
        std::unique_ptr<MNN::AttributeT> strideAttr(new MNN::AttributeT);
        strideAttr->key = "stride";
        strideAttr->i = 1;
        extra->attr.emplace_back(std::move(strideAttr));
    }
}

REGISTER_CONVERTER(StridedSliceTorch, slice);
