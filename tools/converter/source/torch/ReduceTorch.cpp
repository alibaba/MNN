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
        {"sum", MNN::ReductionType_SUM},
        {"mean", MNN::ReductionType_MEAN},
        {"max_reduce", MNN::ReductionType_MAXIMUM},
        {"min_reduce", MNN::ReductionType_MINIMUM},
        {"prod", MNN::ReductionType_PROD},
        {"all", MNN::ReductionType_ALL},
        {"any", MNN::ReductionType_ANY},
    };
    const auto inputs = node->inputs();
    auto param = new MNN::ReductionParamT;
    std::string opType = getRealOpType(node);
    param->operation = gMaps[opType];

    if (opType == "mean") {
        const auto dims = getValue<std::vector<int64_t>>(inputs[1]);
        for (int i : dims) {
            param->dim.push_back(i);
        }
        param->keepDims = getValue<bool>(inputs[2]);
    } else {
        const auto dim = getValue<int64_t>(inputs[1]);
        param->dim.push_back(dim);
    }
    if (dstOp->outputIndexes.size() > 1) {
        int realOutput = dstOp->outputIndexes[0];
        dstOp->outputIndexes.clear();
        dstOp->outputIndexes.push_back(realOutput);
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(ReduceTorch, sum);
REGISTER_CONVERTER(ReduceTorch, mean);
REGISTER_CONVERTER(ReduceTorch, max_reduce);
REGISTER_CONVERTER(ReduceTorch, min_reduce);
REGISTER_CONVERTER(ReduceTorch, prod);
REGISTER_CONVERTER(ReduceTorch, all);
REGISTER_CONVERTER(ReduceTorch, any);
