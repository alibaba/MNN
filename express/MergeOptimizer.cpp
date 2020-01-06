//
//  MergeOptimizer.cpp
//  MNN
//
//  Created by MNN on 2019/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MergeOptimizer.hpp"
#include <map>
#include "Utils.hpp"
#include "BasicOptimizer_generated.h"
#define FLATBUFFERS_PREFER_PRINTF
#include "flatbuffers/util.h"

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
    std::map<EXPRP, int> varIndexOffset;
    std::vector<VARP> inputs;
    std::unique_ptr<MNN::Optimizer::MergeT> merge(new MNN::Optimizer::MergeT);
    merge->backend.reset(new MNN::Optimizer::BackendConfigT);
    merge->backend->numberThread = mNumberThread;
    merge->backend->type         = (MNN::ForwardType)mType;
    merge->backend->power        = (int)mConfig.power;
    merge->backend->precision    = (int)mConfig.precision;
    merge->backend->memroy       = (int)mConfig.memory;

    int tensorOffset = 0;
    for (int i = 0; i < sequence.size(); ++i) {
        auto expr      = sequence[i];
        if (nullptr != expr->get() && OpType_Extra == expr->get()->type()) {
            return true;
        }
        varIndexOffset[expr] = tensorOffset;
        tensorOffset += expr->outputSize();
        if (nullptr == expr->get()) {
            if (expr->inputType() == VARP::INPUT) {
                inputs.emplace_back(Variable::create(expr));
                merge->inputIndexes.emplace_back(varIndexOffset[expr]);
            } else {
                std::unique_ptr<OpT> op;
                VARP var = Variable::create(expr);
                auto& info = *(var->getInfo());
                auto blob        = new BlobT;
                blob->dataFormat = (MNN_DATA_FORMAT)Utils::convertFormat(info.order);
                blob->dims       = info.dim;
                if (info.type.code == halide_type_float) {
                    blob->dataType = DataType_DT_FLOAT;
                    blob->float32s.resize(info.size);
                    ::memcpy(blob->float32s.data(), info.ptr, info.size * sizeof(float));
                } else if (info.type.code == halide_type_int) {
                    blob->dataType = DataType_DT_INT32;
                    blob->int32s.resize(info.size);
                    ::memcpy(blob->int32s.data(), info.ptr, info.size * sizeof(int));
                }
                else if (info.type.code == halide_type_uint && info.type.bits == 8) {
                    blob->dataType = DataType_DT_UINT8;
                    blob->uint8s.resize(info.size);
                    ::memcpy(blob->uint8s.data(), info.ptr, info.size * sizeof(uint8_t));
                }
                op.reset(new OpT);
                op->type       = OpType_Const;
                op->main.type  = OpParameter_Blob;
                op->main.value = blob;
                op->outputIndexes = {varIndexOffset[expr]};
                merge->oplists.emplace_back(std::move(op));
            }
        }
    }
    merge->tensorNumber = tensorOffset;
    for (auto expr : sequence) {
        if (nullptr == expr->get()) {
            continue;
        }
        std::unique_ptr<OpT> op(expr->get()->UnPack());
        auto outputIndexStart = varIndexOffset[expr];
        op->name = EnumNameOpType(op->type) + flatbuffers::NumToString(outputIndexStart+1);
        op->outputIndexes.resize(expr->outputSize());
        for (int i=0; i<expr->outputSize(); ++i) {
            op->outputIndexes[i] = outputIndexStart + i;
        }
        auto exprinputs       = expr->inputs();
        op->inputIndexes.resize(exprinputs.size());
        for (int i = 0; i < exprinputs.size(); ++i) {
            auto inputExpr = exprinputs[i]->expr();
            op->inputIndexes[i] = varIndexOffset[inputExpr.first] + inputExpr.second;
        }
        merge->oplists.emplace_back(std::move(op));
    }
    for (auto var : outputs) {
        auto expr = var->expr();
        merge->outputIndexes.emplace_back(varIndexOffset[expr.first] + expr.second);
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
        auto name = outputs[i]->name();
        outputs[i]->setExpr(mergeExpr, i);
        outputs[i]->setName(name); // merge expr does not copy mOutputNames, so copy to prevent var's name to be erased
    }
    return true;
}
} // namespace Express
} // namespace MNN
