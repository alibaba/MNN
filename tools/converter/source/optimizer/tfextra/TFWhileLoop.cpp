//
//  TFWhileLoop.cpp
//  MNNConverter
//
//  Created by MNN on 2020/06/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include <unordered_map>
#include <unordered_set>

#include "../../common/Common.hpp"
#include "../../common/Global.hpp"
#include "../SubGraphComplete.hpp"
#include "MNN_generated.h"
#include "TFExtraManager.hpp"

namespace MNN {
namespace Express {

class WhileLoopTransform : public TFExtraManager::Transform {
public:
    EXPRP onExecute(EXPRP expr) const override;

private:
    void TransformAliases(MNN::WhileParamT* while_param) const;
    const Attribute* FindAttributeByName(const std::string& name) const;

    void UpdateSubGraphSignatures(const std::unordered_map<std::string, VARP>& inputs, SubGraphProtoT* subgraph) const;

    void FixupBodyGraphOutputs(SubGraphProtoT* body_graph, MNN::WhileParamT* while_param) const;

private:
    mutable std::vector<const Attribute*> attributes_;

    using StringVec = std::vector<std::string>;
};

EXPRP WhileLoopTransform::onExecute(EXPRP expr) const {
    // Clear stateful members.
    attributes_.clear();

    const Op* op = expr->get();
    for (const auto& attr : *(op->main_as_Extra()->attr())) {
        attributes_.push_back(attr);
    }
    auto* while_param = new MNN::WhileParamT;
    std::string cond_graph_name, body_graph_name;

    auto* cond_graph_attr   = FindAttributeByName("cond_graph");
    cond_graph_name         = cond_graph_attr->s()->str();
    auto* body_graph_attr   = FindAttributeByName("body_graph");
    body_graph_name         = body_graph_attr->s()->str();
    while_param->cond_graph = cond_graph_name;
    while_param->body_graph = body_graph_name;

    TransformAliases(while_param);

    std::unordered_map<std::string, VARP> subgraph_inputs;
    for (int idx = 0; idx < while_param->aliases_inputs.size(); ++idx) {
        for (const auto& input : while_param->aliases_inputs[idx]->data) {
            subgraph_inputs.emplace(input, expr->inputs().at(idx));
        }
    }
    auto* ctx = Global<OptimizeContext>::Get();
    MNN_ASSERT(ctx != nullptr);
    std::vector<SubGraphProtoT*>& subgraphs = ctx->subgraphs;
    auto* cond_graph                        = FindSubGraphByName(subgraphs, cond_graph_name);
    auto* body_graph                        = FindSubGraphByName(subgraphs, body_graph_name);
    MNN_ASSERT(cond_graph != nullptr);
    MNN_ASSERT(body_graph != nullptr);
    UpdateSubGraphSignatures(subgraph_inputs, cond_graph);
    UpdateSubGraphSignatures(subgraph_inputs, body_graph);

    // Optimize subgraph and complete op conversion.
    CompleteSubGraph(subgraph_inputs, cond_graph);
    CompleteSubGraph(subgraph_inputs, body_graph);

    const SubGraphProtoT* completed_cond_graph = FindSubGraphByName(ctx->completed_subgraphs, cond_graph_name);
    SubGraphProtoT* completed_body_graph       = FindSubGraphByName(ctx->completed_subgraphs, body_graph_name);
    FixupBodyGraphOutputs(completed_body_graph, while_param);

    //    SubGraphLibrary::Add(completed_cond_graph);
    //    SubGraphLibrary::Add(completed_body_graph);

    std::unique_ptr<MNN::OpT> while_op(new MNN::OpT);
    while_op->type       = OpType_While;
    while_op->name       = op->name()->str();
    while_op->main.type  = OpParameter_WhileParam;
    while_op->main.value = while_param;
    auto while_expr      = Expr::create(while_op.get(), expr->inputs(), expr->outputSize());
    // Set output names.
    int outputSize = expr->outputSize();
    for (int i = 0; i < outputSize; ++i) {
        auto while_var = Variable::create(while_expr, i);
        while_var->setName(expr->outputName(i));
    }
    return std::move(while_expr);
}

void WhileLoopTransform::FixupBodyGraphOutputs(SubGraphProtoT* body_graph, MNN::WhileParamT* while_param) const {
    std::unordered_map<std::string, int> indices;
    for (int i = 0; i < body_graph->tensors.size(); ++i) {
        const std::string& name = body_graph->tensors.at(i);
        indices.emplace(name, i);
    }
    std::unordered_set<int> outputs;
    for (int i : body_graph->outputs) {
        outputs.insert(i);
    }
    auto TryToAppendOutput = [&](const std::string& output) {
        auto it = indices.find(output);
        if (it == indices.end()) {
            return;
        }
        if (outputs.insert(it->second).second) {
            body_graph->outputs.push_back(it->second);
        }
    };
    for (const auto& output : while_param->aliases_outputs) {
        TryToAppendOutput(output);
    }
    for (const auto& updates : while_param->aliases_updates) {
        if (updates->data.size() < 2) {
            continue;
        }
        TryToAppendOutput(updates->data.at(0));
    }
}

const Attribute* WhileLoopTransform::FindAttributeByName(const std::string& name) const {
    for (const auto* attr : attributes_) {
        if (attr->key()->str() == name) {
            return attr;
        }
    }
    return nullptr;
}

void WhileLoopTransform::TransformAliases(MNN::WhileParamT* while_param) const {
    auto* aliases_inputs_attr = FindAttributeByName("aliases_inputs");
    auto* inputs_func         = aliases_inputs_attr->func();
    MNN_ASSERT(inputs_func != nullptr);
    while_param->aliases_inputs.resize(inputs_func->attr()->size());
    for (const auto& it : *(inputs_func->attr())) {
        MNN_ASSERT(it != nullptr);
        int idx = atoi(it->key()->str().c_str());
        while_param->aliases_inputs[idx].reset(new StringVecT);
        MNN_ASSERT(it->list() != nullptr);
        MNN_ASSERT(it->list()->s() != nullptr);
        for (const auto& mapping_name : *(it->list()->s())) {
            while_param->aliases_inputs[idx]->data.push_back(mapping_name->str());
        }
    }

    auto* aliases_outputs_attr = FindAttributeByName("aliases_outputs");
    auto* outputs_func         = aliases_outputs_attr->func();
    MNN_ASSERT(outputs_func != nullptr);
    while_param->aliases_outputs.resize(outputs_func->attr()->size());
    for (const auto& it : *(outputs_func->attr())) {
        MNN_ASSERT(it != nullptr);
        int idx = atoi(it->key()->str().c_str());
        MNN_ASSERT(it->list() != nullptr);
        MNN_ASSERT(it->list()->s() != nullptr);
        MNN_ASSERT(it->list()->s()->size() == 1);
        while_param->aliases_outputs[idx] = it->list()->s()->Get(0)->str();
    }

    auto* aliases_updates_attr = FindAttributeByName("aliases_updates");
    auto* updates_func         = aliases_updates_attr->func();
    while_param->aliases_updates.resize(updates_func->attr()->size());
    int idx = 0;
    for (const auto& it : *(updates_func->attr())) {
        MNN_ASSERT(it != nullptr);
        while_param->aliases_updates[idx].reset(new StringVecT);
        while_param->aliases_updates[idx]->data.push_back(it->key()->str());

        MNN_ASSERT(it->list() != nullptr);
        MNN_ASSERT(it->list()->s() != nullptr);
        MNN_ASSERT(it->list()->s()->size() == 1);
        const auto& mapping_name = it->list()->s()->Get(0)->str();
        while_param->aliases_updates[idx]->data.push_back(mapping_name);
        ++idx;
    }
}

void WhileLoopTransform::UpdateSubGraphSignatures(const std::unordered_map<std::string, VARP>& inputs,
                                                  SubGraphProtoT* subgraph) const {
    for (auto& node : subgraph->nodes) {
        if (node->type != OpType_Input) {
            continue;
        }
        int idx           = node->outputIndexes.at(0);
        std::string input = subgraph->tensors.at(idx);
        auto iter = inputs.find(input);
        if (iter == inputs.end()) {
            continue;
        }
        VARP input_var    = iter->second;
        auto* info        = input_var->getInfo();
        if (info != nullptr) {
            InputT* param  = node->main.AsInput();
            param->dims    = info->dim;
            param->dtype   = convertDataType(info->type);
            param->dformat = convertFormat(info->order);
        }
    }
}

static auto gRegister = []() {
    TFExtraManager::get()->insert("CustomWhileLoop",
                                  std::shared_ptr<TFExtraManager::Transform>(new WhileLoopTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
