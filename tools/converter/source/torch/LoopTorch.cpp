//
//  LoopTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/07/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(LoopTorch);

MNN::OpType LoopTorch::opType() {
    return MNN::OpType_While;
}
MNN::OpParameter LoopTorch::type() {
    return MNN::OpParameter_WhileParam;
}
std::vector<int> LoopTorch::inputTensorIdx() {
    return {-1};
}

void LoopTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::WhileParamT;
    const auto bodyBlock = node->blocks()[0];
    param->cond_graph = dstOp->name + "/cond";
    param->body_graph = dstOp->name + "/body";
    // loop: for (int i = 0; i < M && keep_going; i++)
    // i - iName; mName - M; kName - keep_going
    std::string iName = bodyBlock->inputs().at(0)->debugName(),
                mName = node->input(0)->debugName(),
                kName = node->input(1)->debugName();
    // declare int i = 0;
    int idxI = scope->buildIntConstOp({0}, iName);
    if (std::find(dstOp->inputIndexes.begin(), dstOp->inputIndexes.end(), idxI) == dstOp->inputIndexes.end()) {
        dstOp->inputIndexes.push_back(idxI);
    }
    // build cond
    scope->buildCondGraph(param->cond_graph, iName, mName, kName);
    // build body
    scope->buildSubGraph(bodyBlock, param->body_graph, true);
    scope->dealSubgraphDepsForOp(dstOp);

    for (int idx : dstOp->inputIndexes) {
        std::unique_ptr<MNN::StringVecT> inputT(new MNN::StringVecT);
        inputT->data.emplace_back(scope->lookupTensorByIdx(idx));
        param->aliases_inputs.emplace_back(std::move(inputT));
    }
    // update block[0]->outputs: [ keep_going, user_def_vars... ]
    for (int i = 0; i < bodyBlock->outputs().size(); i++) {
        std::unique_ptr<MNN::StringVecT> updateT(new MNN::StringVecT);
        auto bodyOutput = bodyBlock->outputs().at(i)->debugName();
        updateT->data.emplace_back(bodyOutput);
        updateT->data.emplace_back(node->inputs().at(1 + i)->debugName());
        param->aliases_updates.emplace_back(std::move(updateT));
        if (i > 0) {
            param->aliases_outputs.push_back(bodyOutput);
        }
    }
    // update i
    std::unique_ptr<MNN::StringVecT> updateT(new MNN::StringVecT);
    updateT->data.emplace_back(param->body_graph + "/increment_i");
    updateT->data.emplace_back(iName);
    param->aliases_updates.emplace_back(std::move(updateT));
    dstOp->main.value = param;
}

REGISTER_CONVERTER(LoopTorch, Loop);
