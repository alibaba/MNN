//
//  LoopOnnx.cpp
//  MNN
//
//  Created by MNN on 2021/06/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(LoopOnnx);

MNN::OpType LoopOnnx::opType() {
    return MNN::OpType_While;
}
MNN::OpParameter LoopOnnx::type() {
    return MNN::OpParameter_WhileParam;
}

void LoopOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                   OnnxScope* scope) {
    auto param = new MNN::WhileParamT;
    dstOp->name += "/Loop";
    param->cond_graph = dstOp->name +  "/cond";
    param->body_graph = dstOp->name +  "/body";
    auto body = &onnxNode->attribute(0).g();
    // loop: for (int i = 0; i < M && keep_going; i++)
    // i - iName; mName - M; kName - keep_going; uName - user_defined_val
    std::string iName = body->input(0).name(),
                mName = onnxNode->input(0),
                kName = onnxNode->input(1),
                uName = onnxNode->output(0) + "/tensorArray";
    // declare int i = 0;
    int idxI = scope->buildIntConstOp({0}, iName);
    if (std::find(dstOp->inputIndexes.begin(), dstOp->inputIndexes.end(), idxI) == dstOp->inputIndexes.end()) {
        dstOp->inputIndexes.push_back(idxI);
    }
    // declare user_defined_val
    auto idxU = scope->buildTensorArrayOp({1}, true, uName);
    if (std::find(dstOp->inputIndexes.begin(), dstOp->inputIndexes.end(), idxI) == dstOp->inputIndexes.end()) {
        dstOp->inputIndexes.push_back(idxU.first);
        dstOp->inputIndexes.push_back(idxU.second);
    }
    // build cond
    scope->buildCondGraph(param->cond_graph, iName, mName, kName);
    // build body
    scope->buildSubGraph(body, param->body_graph, uName, true);
    scope->dealSubgraphDepsForOp(dstOp);
    for (int idx : dstOp->inputIndexes) {
        std::unique_ptr<MNN::StringVecT> inputT(new MNN::StringVecT);
        inputT->data.emplace_back(scope->lookupTensorByIdx(idx));
        param->aliases_inputs.emplace_back(std::move(inputT));
    }
    // update body.outputs: [ keep_going, user_def_vars... ]
    for (int i = 0; i < body->output_size(); i++) {
        auto bodyOutput = body->output(i).name();
        // update [keep_going, user_def_vars...]
        if (i + 1 < onnxNode->input_size()) {
            std::unique_ptr<MNN::StringVecT> updateT(new MNN::StringVecT);
            updateT->data.emplace_back(bodyOutput);
            updateT->data.emplace_back(onnxNode->input(i + 1));
            param->aliases_updates.emplace_back(std::move(updateT));
        }
    }
    // update i
    std::unique_ptr<MNN::StringVecT> updateI(new MNN::StringVecT);
    updateI->data.emplace_back(param->body_graph + "/increment_i");
    updateI->data.emplace_back(iName);
    param->aliases_updates.emplace_back(std::move(updateI));
    // update user_defined_vals
    std::unique_ptr<MNN::StringVecT> updateU(new MNN::StringVecT);
    updateU->data.emplace_back(param->body_graph + "/accumulate_u");
    updateU->data.emplace_back(uName);
    param->aliases_updates.emplace_back(std::move(updateU));
    dstOp->main.value = param;
    param->aliases_outputs.push_back(param->body_graph + "/accumulate_u");
}

REGISTER_CONVERTER(LoopOnnx, Loop);
