//
//  MergeOptimizer.cpp
//  MNN
//
//  Created by MNN on 2019/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MergeOptimizer.hpp"
#include <map>
#include "BasicOptimizer_generated.h"
namespace MNN {
namespace Express {

MergeOptimizer::MergeOptimizer(MNNForwardType type, int numberThread, BackendConfig* config) {
    if (nullptr != config) {
        mConfig = *config;
    }
    mType         = type;
    mNumberThread = numberThread;
}

Optimizer::Cost MergeOptimizer::onMeasure(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters) {
    Cost cost;
    cost.compute = 0.0f;
    cost.memory  = 0.0f;
    return cost;
}
bool MergeOptimizer::onExecute(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters) {
    auto sequence = Variable::getExecuteOrder(outputs);
    if (1 == sequence.size()) {
        return true;
    }
    std::map<EXPRP, std::vector<int>> worked;
    std::map<VARP, int> varIndex;
    std::vector<EXPRP> queue;
    std::vector<VARP> inputs;
    std::unique_ptr<MNN::Optimizer::MergeT> merge(new MNN::Optimizer::MergeT);
    queue.reserve(sequence.size());
    merge->tensorNumber = sequence.size();
    merge->backend.reset(new MNN::Optimizer::BackendConfigT);
    merge->backend->numberThread = mNumberThread;
    merge->backend->type         = (MNN::ForwardType)mType;
    merge->backend->power        = (int)mConfig.power;
    merge->backend->precision    = (int)mConfig.precision;
    merge->backend->memroy       = (int)mConfig.memory;

    for (int i = 0; i < sequence.size(); ++i) {
        auto var      = sequence[i];
        varIndex[var] = i;
        if (var->expr().first->get()->type() == OpType_Input) {
            inputs.emplace_back(var);
        }
        auto exprInfo     = var->expr();
        if (exprInfo.first->get()->type() == OpType_Input) {
            merge->inputIndexes.emplace_back(i);
            continue;
        }
        if (worked.find(exprInfo.first) != worked.end()) {
            worked[exprInfo.first][exprInfo.second] = i;
            continue;
        }
        worked.insert(std::make_pair(exprInfo.first, std::vector<int>(exprInfo.first->outputSize())));
        worked[exprInfo.first][exprInfo.second] = i;
        queue.emplace_back(exprInfo.first);
    }
    for (auto expr : queue) {
        std::unique_ptr<OpT> op(expr->get()->UnPack());
        op->outputIndexes = worked[expr];
        auto exprinputs       = expr->inputs();
        op->inputIndexes.resize(exprinputs.size());
        for (int i = 0; i < exprinputs.size(); ++i) {
            op->inputIndexes[i] = varIndex[exprinputs[i]];
        }
        merge->oplists.emplace_back(std::move(op));
    }
    for (auto var : outputs) {
        merge->outputIndexes.emplace_back(varIndex[var]);
    }

    std::unique_ptr<OpT> mergeOp(new OpT);
    mergeOp->type       = OpType_Extra;
    mergeOp->name       = outputs[0]->name();
    mergeOp->main.type  = OpParameter_Extra;
    mergeOp->main.value = new ExtraT;
    auto plugin         = mergeOp->main.AsExtra();
    plugin->type        = "Session";
    plugin->engine      = "MNN";

    flatbuffers::FlatBufferBuilder builder;
    auto offset = MNN::Optimizer::Merge::Pack(builder, merge.get());
    builder.Finish(offset);
    plugin->info.resize(builder.GetSize());
    ::memcpy(plugin->info.data(), builder.GetBufferPointer(), builder.GetSize());

    auto mergeExpr = Expr::create(mergeOp.get(), inputs, (int)outputs.size());
    mergeExpr->setName(outputs[0]->name());
    for (int i = 0; i < outputs.size(); ++i) {
        Variable::setExpr(outputs[i], mergeExpr, i);
    }
    return true;
}
} // namespace Express
} // namespace MNN
