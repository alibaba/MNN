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
static MNN_DATA_FORMAT _convertFormat(Dimensionformat format) {
    static std::map<Dimensionformat, MNN_DATA_FORMAT> gMap = {
        {NCHW, MNN_DATA_FORMAT_NCHW}, {NHWC, MNN_DATA_FORMAT_NHWC}, {NC4HW4, MNN_DATA_FORMAT_NC4HW4}};
    return gMap[format];
}
MergeOptimizer::MergeOptimizer(MNNForwardType type, int numberThread, BackendConfig* config) {
    if (nullptr != config) {
        mConfig = *config;
    }
    mType         = type;
    mNumberThread = numberThread;
}

Optimizer::Cost MergeOptimizer::onMeasure(const Model& model, std::shared_ptr<Parameters> parameters) {
    Cost cost;
    cost.compute = 0.0f;
    cost.memory  = 0.0f;
    return cost;
}
bool MergeOptimizer::onExecute(Model& model, std::shared_ptr<Parameters> parameters) {
    for (auto var : model.inputs) {
        if (nullptr == var->getInfo()) {
            MNN_ERROR("Input not ready, please resize input firstly\n");
            return false;
        }
    }
    std::map<EXPRP, std::vector<int>> worked;
    std::map<VARP, int> varIndex;
    std::vector<EXPRP> queue;
    std::unique_ptr<MNN::Optimizer::MergeT> merge(new MNN::Optimizer::MergeT);
    queue.reserve(model.sequence.size());
    merge->tensors.resize(model.sequence.size());
    merge->backend.reset(new MNN::Optimizer::BackendConfigT);
    merge->backend->numberThread = mNumberThread;
    merge->backend->type         = (MNN::ForwardType)mType;
    merge->backend->power        = (int)mConfig.power;
    merge->backend->precision    = (int)mConfig.precision;
    merge->backend->memroy       = (int)mConfig.memory;

    for (int i = 0; i < model.sequence.size(); ++i) {
        auto var      = model.sequence[i];
        varIndex[var] = i;
        auto info     = var->getInfo();
        if (nullptr == info) {
            MNN_ERROR("Get info error when optimize\n");
            return false;
        }
        std::unique_ptr<BlobT> blob(new BlobT);
        blob->dataFormat  = _convertFormat(info->order);
        blob->dims        = info->dim;
        blob->dataType    = DataType_DT_FLOAT;
        merge->tensors[i] = std::move(blob);
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
        auto inputs       = expr->inputs();
        op->inputIndexes.resize(inputs.size());
        for (int i = 0; i < inputs.size(); ++i) {
            op->inputIndexes[i] = varIndex[inputs[i]];
        }
        merge->oplists.emplace_back(std::move(op));
    }
    for (auto var : model.outputs) {
        merge->outputIndexes.emplace_back(varIndex[var]);
    }

    std::unique_ptr<OpT> mergeOp(new OpT);
    mergeOp->type       = OpType_PLUGIN;
    mergeOp->name       = model.outputs[0]->name();
    mergeOp->main.type  = OpParameter_Plugin;
    mergeOp->main.value = new PluginT;
    auto plugin         = mergeOp->main.AsPlugin();
    plugin->type        = "Merge";
    plugin->buffer.resize(1);
    plugin->buffer[0].reset(new BlobT);
    plugin->buffer[0]->dataType = DataType_DT_UINT8;

    flatbuffers::FlatBufferBuilder builder;
    auto offset = MNN::Optimizer::Merge::Pack(builder, merge.get());
    builder.Finish(offset);
    plugin->buffer[0]->dims = {(int)builder.GetSize()};
    plugin->buffer[0]->uint8s.resize(builder.GetSize());
    ::memcpy(plugin->buffer[0]->uint8s.data(), builder.GetBufferPointer(), builder.GetSize());

    auto mergeExpr = Expr::create(std::move(mergeOp), model.inputs, (int)model.outputs.size());
    for (int i = 0; i < model.outputs.size(); ++i) {
        auto name = model.outputs[i]->name();
        model.outputs[i] = Variable::create(mergeExpr, i);
        model.outputs[i]->setName(name);
    }
    model.reorder();
    return true;
}
} // namespace Express
} // namespace MNN
