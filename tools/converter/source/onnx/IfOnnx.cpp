//
//  IfOnnx.cpp
//  MNN
//
//  Created by MNN on 2021/06/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"
#include <MNN/MNNDefine.h>
DECLARE_OP_CONVERTER(IfOnnx);

MNN::OpType IfOnnx::opType() {
    return MNN::OpType_If;
}
MNN::OpParameter IfOnnx::type() {
    return MNN::OpParameter_IfParam;
}

void IfOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                   OnnxScope* scope) {
    auto param = new MNN::IfParamT;
    const ::onnx::GraphProto *thenG = nullptr, *elseG = nullptr;
    for (const auto& attr : onnxNode->attribute()) {
        if (attr.name() == "then_branch") {
            thenG = &attr.g();
            param->then_graph = thenG->name();
        } else if (attr.name() == "else_branch") {
            elseG = &attr.g();
            param->else_graph = elseG->name();
        }
    }
    if (thenG == nullptr || elseG == nullptr) {
        MNN_ERROR("Invalid Attrs, then_branch and else_branch is required\n");
        return;
    }
    auto dealWithSubGraph = [=](const ::onnx::GraphProto* graph, std::string& name) {
        std::vector<std::string> inputs, outputs(graph->output_size());
        for (const auto& input : graph->input()) {
            const auto& inits = graph->initializer();
            auto iter = std::find_if(inits.begin(), inits.end(), [&](const ::onnx::TensorProto& p) { return p.name() == input.name(); });
            if (iter == inits.end()) {
                inputs.push_back(input.name());
            }
        }
        auto implicitInputs = scope->buildSubGraph(graph, name, false);
        inputs.insert(inputs.end(), implicitInputs.begin(), implicitInputs.end());
        std::transform(graph->output().begin(), graph->output().end(), outputs.begin(), [](const ::onnx::ValueInfoProto& p) { return p.name(); });
        return std::make_pair(inputs, outputs);
    };
    std::pair<std::vector<std::string>, std::vector<std::string>> thenInOuts, elseInOuts;
    MNN_ASSERT(thenG->node_size() > 0 || elseG->node_size() > 0);
    if (thenG->node_size() > 0) {
        thenInOuts = dealWithSubGraph(thenG, param->then_graph);
    }
    if (elseG->node_size() > 0) {
        elseInOuts = dealWithSubGraph(elseG, param->else_graph);
    }
    if (thenG->node_size() == 0) {
        thenInOuts = elseInOuts;
        param->then_graph = param->else_graph;
    } else if (elseG->node_size() == 0) {
        elseInOuts = thenInOuts;
        param->else_graph = param->then_graph;
    }
    auto thenInputs = thenInOuts.first, thenOutputs = thenInOuts.second;
    auto elseInputs = elseInOuts.first, elseOutputs = elseInOuts.second;
    
    bool sameOutput = (thenOutputs.size() == elseOutputs.size() && thenOutputs.size() == onnxNode->output_size());
    if (!sameOutput) {
        MNN_ERROR("Op(If) and its subgraphs (then_branch, else_branch) must have same output number\n");
        return;
    }
    for (int i = 0; i < onnxNode->output_size(); ++i) {
        std::unique_ptr<MNN::StringVecT> pair(new MNN::StringVecT);
        pair->data.assign({thenOutputs[i], elseOutputs[i]});
        param->aliases_outputs.emplace_back(std::move(pair));
    }
    auto mergeInputs = thenInputs;
    for (const auto& name : elseInputs) {
        if (std::find(thenInputs.begin(), thenInputs.end(), name) == thenInputs.end()) {
            mergeInputs.push_back(name);
        }
    }
    { // cond input
        std::unique_ptr<MNN::StringVecT> pair(new MNN::StringVecT);
        param->aliases_inputs.emplace_back(std::move(pair));
    }
    for (const auto& name : mergeInputs) {
        std::unique_ptr<MNN::StringVecT> pair(new MNN::StringVecT);
        pair->data.emplace_back(name);
        param->aliases_inputs.emplace_back(std::move(pair));
        // subgraph own by IF may introduce extra input which is not exist on current graph, create corresponding input op here
        scope->addInputForOp(dstOp, name, true);
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(IfOnnx, If);
