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

void ReduceTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    static std::map<std::string, MNN::ReductionType> gMaps{
        {"sum_reduce", MNN::ReductionType_SUM},
        {"mean", MNN::ReductionType_MEAN},
        {"max_reduce", MNN::ReductionType_MAXIMUM},
        {"amax", MNN::ReductionType_MAXIMUM},
        {"min_reduce", MNN::ReductionType_MINIMUM},
        {"amin", MNN::ReductionType_MINIMUM},
        {"prod", MNN::ReductionType_PROD},
        {"all", MNN::ReductionType_ALL},
        {"any", MNN::ReductionType_ANY},
    };
    const auto inputs = node->inputs();
    auto param = new MNN::ReductionParamT;
    std::string opType = getRealOpType(node);
    param->operation = gMaps[opType];

    if (opType == "sum_reduce" || opType == "mean" || opType == "amax" || opType == "amin") {
        const auto dims = getValue<std::vector<int64_t>>(inputs[1]);
        for (int i : dims) {
            param->dim.push_back(i);
        }
        param->keepDims = getValue<bool>(inputs[2]);
    } else {
        if (inputs[1]->type()->kind() == c10::TypeKind::IntType) {
            const auto dim = getValue<int64_t>(inputs[1]);
            param->dim.push_back(dim);
        } else {
            const auto dims = getValue<std::vector<int64_t>>(inputs[1]);
            for (auto dim : dims) {
                param->dim.push_back(dim);
            }
        }
    }
    if (dstOp->outputIndexes.size() > 1) {
        int realOutput = dstOp->outputIndexes[0];
        dstOp->outputIndexes.clear();
        dstOp->outputIndexes.push_back(realOutput);
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(ReduceTorch, sum_reduce);
REGISTER_CONVERTER(ReduceTorch, mean);
REGISTER_CONVERTER(ReduceTorch, max_reduce);
REGISTER_CONVERTER(ReduceTorch, min_reduce);
REGISTER_CONVERTER(ReduceTorch, prod);
REGISTER_CONVERTER(ReduceTorch, all);
REGISTER_CONVERTER(ReduceTorch, any);
REGISTER_CONVERTER(ReduceTorch, amin);
REGISTER_CONVERTER(ReduceTorch, amax);

DECLARE_OP_CONVERTER(NormTorch);

MNN::OpType NormTorch::opType() {
    return MNN::OpType_Extra;
}
MNN::OpParameter NormTorch::type() {
    return MNN::OpParameter_Extra;
}
std::vector<int> NormTorch::inputTensorIdx() {
    return {0};
}

void NormTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto extra = new MNN::ExtraT;
    dstOp->main.value = extra;
    extra->engine     = "Torch";
    extra->type       = "norm";
    auto type = getRealOpType(node);
    extra->attr.resize(3);
    extra->attr[0].reset(new MNN::AttributeT);
    extra->attr[0]->key = "ord";
    extra->attr[1].reset(new MNN::AttributeT);
    extra->attr[1]->key = "dim";
    extra->attr[2].reset(new MNN::AttributeT);
    extra->attr[2]->key = "keepDim";
    if (type == "frobenius_norm") {
        extra->attr[0]->i = 2;
        auto dims = getValue<std::vector<int64_t>>(node->input(1));
        extra->attr[1]->i = dims[0];
        extra->attr[2]->i = getValue<bool>(node->input(2));
    } else {
        auto ord = node->input(1);
        auto kind = ord->type()->kind();
        if (kind == c10::TypeKind::FloatType) {
            extra->attr[0]->i = getValue<double>(node->input(1));
        } else {
            extra->attr[0]->i = getValue<int64_t>(node->input(1));
        }
        auto dims = getValue<std::vector<int64_t>>(node->input(2));
        extra->attr[1]->i = dims[0];
        extra->attr[2]->i = getValue<bool>(node->input(3));
    }
}

REGISTER_CONVERTER(NormTorch, norm);
REGISTER_CONVERTER(NormTorch, frobenius_norm);
