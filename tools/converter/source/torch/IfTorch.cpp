//
//  IfTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(IfTorch);

MNN::OpType IfTorch::opType() {
    return MNN::OpType_If;
}
MNN::OpParameter IfTorch::type() {
    return MNN::OpParameter_IfParam;
}
std::vector<int> IfTorch::inputTensorIdx() {
    return {0};
}

void IfTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::IfParamT;
    const auto blocks = node->blocks();
    param->then_graph = dstOp->name + "/then";
    param->else_graph = dstOp->name + "/else";
    // then
    scope->buildSubGraph(blocks[0], param->then_graph);
    // else
    scope->buildSubGraph(blocks[1], param->else_graph);
    scope->dealSubgraphDepsForOp(dstOp);
    for (int idx : dstOp->inputIndexes) {
        std::unique_ptr<MNN::StringVecT> inputT(new MNN::StringVecT);
        inputT->data.emplace_back(scope->lookupTensorByIdx(idx));
        param->aliases_inputs.emplace_back(std::move(inputT));
    }
    std::unique_ptr<MNN::StringVecT> outputPari(new MNN::StringVecT);
    outputPari->data.emplace_back(blocks[0]->outputs()[0]->debugName());
    outputPari->data.emplace_back(blocks[1]->outputs()[0]->debugName());
    param->aliases_outputs.emplace_back(std::move(outputPari));
    dstOp->main.value = param;
}

REGISTER_CONVERTER(IfTorch, If);
