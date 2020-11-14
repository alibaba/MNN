//
//  TFCondition.cpp
//  MNNConverter
//
//  Created by MNN on 2020/06/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>

#include "../../common/Common.hpp"
#include "../../common/Global.hpp"
#include "../SubGraphComplete.hpp"
#include "MNN_generated.h"
#include "TFExtraManager.hpp"

namespace MNN {
namespace Express {

class ConditionTransform : public TFExtraManager::Transform {
public:
    EXPRP onExecute(EXPRP expr) const override;

private:
    void TransformAliases(MNN::IfParamT* if_param) const;
    const Attribute* FindAttributeByName(const std::string& name) const;

    void UpdateSubGraphSignatures(const std::unordered_map<std::string, VARP>& inputs, SubGraphProtoT* subgraph) const;

private:
    mutable std::vector<const Attribute*> attributes_;

    using StringVec = std::vector<std::string>;
    mutable std::vector<StringVec> aliases_inputs_;
    mutable std::vector<StringVec> aliases_outputs_;
};

EXPRP ConditionTransform::onExecute(EXPRP expr) const {
    // Clear stateful members.
    attributes_.clear();
    aliases_inputs_.clear();
    aliases_outputs_.clear();

    const Op* op = expr->get();
    for (const auto& attr : *(op->main_as_Extra()->attr())) {
        attributes_.push_back(attr);
    }
    auto* if_param = new MNN::IfParamT;
    std::string then_graph_name, else_graph_name;

    auto* then_graph_attr = FindAttributeByName("then_graph");
    then_graph_name       = then_graph_attr->s()->str();
    auto* else_graph_attr = FindAttributeByName("else_graph");
    else_graph_name       = else_graph_attr->s()->str();
    if_param->then_graph  = then_graph_name;
    if_param->else_graph  = else_graph_name;

    TransformAliases(if_param);

    std::unordered_map<std::string, VARP> subgraph_inputs;
    for (int idx = 0; idx < aliases_inputs_.size(); ++idx) {
        for (const auto& input : aliases_inputs_.at(idx)) {
            subgraph_inputs.emplace(input, expr->inputs().at(idx));
        }
    }
    auto* ctx = Global<OptimizeContext>::Get();
    MNN_ASSERT(ctx != nullptr);
    std::vector<SubGraphProtoT*>& subgraphs = ctx->subgraphs;
    auto* then_graph                        = FindSubGraphByName(subgraphs, then_graph_name);
    auto* else_graph                        = FindSubGraphByName(subgraphs, else_graph_name);
    MNN_ASSERT(then_graph != nullptr);
    MNN_ASSERT(else_graph != nullptr);
    UpdateSubGraphSignatures(subgraph_inputs, then_graph);
    UpdateSubGraphSignatures(subgraph_inputs, else_graph);

    // Optimize subgraph and complete op conversion.
    CompleteSubGraph(subgraph_inputs, then_graph);
    CompleteSubGraph(subgraph_inputs, else_graph);

    const SubGraphProtoT* completed_then_graph = FindSubGraphByName(ctx->completed_subgraphs, then_graph_name);
    const SubGraphProtoT* completed_else_graph = FindSubGraphByName(ctx->completed_subgraphs, else_graph_name);
    //    SubGraphLibrary::Add(completed_then_graph);
    //    SubGraphLibrary::Add(completed_else_graph);

    std::unique_ptr<MNN::OpT> if_op(new MNN::OpT);
    if_op->type       = OpType_If;
    if_op->name       = op->name()->str();
    if_op->main.type  = OpParameter_IfParam;
    if_op->main.value = if_param;
    auto if_expr      = Expr::create(if_op.get(), expr->inputs(), expr->outputSize());
    if_expr->setName(expr->name());
    // Set output names.
    int outputSize = expr->outputSize();
    for (int i = 0; i < outputSize; ++i) {
        auto if_var = Variable::create(if_expr, i);
        if_var->setName(expr->outputName(i));
    }
    return std::move(if_expr);
}

const Attribute* ConditionTransform::FindAttributeByName(const std::string& name) const {
    for (const auto* attr : attributes_) {
        if (attr->key()->str() == name) {
            return attr;
        }
    }
    return nullptr;
}

void ConditionTransform::TransformAliases(MNN::IfParamT* if_param) const {
    auto* aliases_inputs_attr = FindAttributeByName("aliases_inputs");
    auto* inputs_func         = aliases_inputs_attr->func();
    MNN_ASSERT(inputs_func != nullptr);
    aliases_inputs_.resize(inputs_func->attr()->size());
    for (const auto& it : *(inputs_func->attr())) {
        MNN_ASSERT(it != nullptr);
        int idx = atoi(it->key()->str().c_str());
        MNN_ASSERT(it->list() != nullptr);
        if (!(it->list()->s())) {
            continue;
        }
        for (const auto& mapping_name : *(it->list()->s())) {
            aliases_inputs_[idx].push_back(mapping_name->str());
        }
    }

    auto* aliases_outputs_attr = FindAttributeByName("aliases_outputs");
    auto* outputs_func         = aliases_outputs_attr->func();
    MNN_ASSERT(outputs_func != nullptr);
    aliases_outputs_.resize(outputs_func->attr()->size());
    for (const auto& it : *(outputs_func->attr())) {
        MNN_ASSERT(it != nullptr);
        int idx = atoi(it->key()->str().c_str());
        MNN_ASSERT(it->list() != nullptr);
        MNN_ASSERT(it->list()->s() != nullptr);
        MNN_ASSERT(it->list()->s()->size() == 2);
        for (const auto& mapping_name : *(it->list()->s())) {
            aliases_outputs_[idx].push_back(mapping_name->str());
        }
    }

    for (const auto& v : aliases_inputs_) {
        MNN::StringVecT* inputs = new MNN::StringVecT;
        for (const auto& mapping_name : v) {
            inputs->data.push_back(mapping_name);
        }
        if_param->aliases_inputs.emplace_back(inputs);
    }
    for (const auto& v : aliases_outputs_) {
        MNN::StringVecT* outputs = new MNN::StringVecT;
        for (const auto& mapping_name : v) {
            outputs->data.push_back(mapping_name);
        }
        if_param->aliases_outputs.emplace_back(outputs);
    }
}

void ConditionTransform::UpdateSubGraphSignatures(const std::unordered_map<std::string, VARP>& inputs,
                                                  SubGraphProtoT* subgraph) const {
    for (auto& node : subgraph->nodes) {
        if (node->type != OpType_Input) {
            continue;
        }
        int idx           = node->outputIndexes.at(0);
        std::string input = subgraph->tensors.at(idx);
        VARP input_var    = inputs.at(input);
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
    TFExtraManager::get()->insert("CustomCondition",
                                  std::shared_ptr<TFExtraManager::Transform>(new ConditionTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
