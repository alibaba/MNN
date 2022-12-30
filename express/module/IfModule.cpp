//
//  IfModule.cpp
//  MNN
//
//  Created by MNN on 2020/09/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "IfModule.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {
static int _findPos(const std::vector<std::string>& names, const std::string& key) {
    for (int i=0; i<names.size(); ++i) {
        if (names[i] == key) {
            return i;
        }
    }
    return -1;
}
std::vector<Express::VARP> IfModule::onForward(const std::vector<Express::VARP>& inputs) {
    std::vector<Express::VARP> outputs(mOutputFromElse.size());
    MNN_ASSERT(mOutputFromThen.size() == mOutputFromElse.size());


    if (inputs[0]->readMap<int>()[0] > 0) {
        std::vector<Express::VARP> subInputs(mInputForThen.size());
        for (auto& p : mInputForThen) {
            subInputs[p.first] = inputs[p.second];
        }
        auto subOutputs = mThen->onForward(subInputs);
        for (int i=0; i<mOutputFromThen.size(); ++i) {
            outputs[i] = subOutputs[mOutputFromThen[i]];
        }
    } else {
        std::vector<Express::VARP> subInputs(mInputForElse.size());
        for (auto& p : mInputForElse) {
            subInputs[p.first] = inputs[p.second];
        }
        auto subOutputs = mElse->onForward(subInputs);
        for (int i=0; i<mOutputFromElse.size(); ++i) {
            outputs[i] = subOutputs[mOutputFromElse[i]];
        }
    }
    return outputs;
}
IfModule* IfModule::create(const Op* op, const std::map<std::string, SubGraph>& subGraph) {
    auto module = new IfModule;
    module->setType("IfModule");
    auto ifParam = op->main_as_IfParam();
    auto& thenG = subGraph.find(ifParam->then_graph()->str())->second;
    auto& elseG = subGraph.find(ifParam->else_graph()->str())->second;
    module->mElse = elseG.m;
    module->mThen = thenG.m;
    if (nullptr != op->name()) {
        module->setName(op->name()->str());
    }


    /** Compute map index
     std::vector<std::pair<int, int>> mInputForThen;

     // First mElse' index, Second: inputs's index
     std::vector<std::pair<int, int>> mInputForElse;

     std::vector<int> mOutputFromThen;
     std::vector<int> mOutputFromElse;
     */
    // Map Inputs
    for (int i=0; i<ifParam->aliases_inputs()->size(); ++i) {
        auto index = i;
        auto data = ifParam->aliases_inputs()->GetAs<StringVec>(i);
        if (nullptr == data->data()) {
            continue;
        }
        for (int s=0; s<data->data()->size(); ++s) {
            auto name = data->data()->GetAsString(s)->str();
            auto thenPos = _findPos(thenG.inputs, name);
            if (thenPos >= 0) {
                module->mInputForThen.emplace_back(std::make_pair(thenPos, i));
            }
            auto elsePos = _findPos(elseG.inputs, name);
            if (elsePos >= 0) {
                module->mInputForElse.emplace_back(std::make_pair(elsePos, i));
            }
        }
    }
    MNN_ASSERT(module->mInputForElse.size() == elseG.inputs.size());
    MNN_ASSERT(module->mInputForThen.size() == thenG.inputs.size());
    // Map outputs
    auto output = ifParam->aliases_outputs();
    if (output == nullptr) { // Onnx
        for (int i = 0; i < op->outputIndexes()->size(); ++i) {
            module->mOutputFromThen.push_back(i);
            module->mOutputFromElse.push_back(i);
        }
        return module;
    }
    module->mOutputFromThen.resize(output->size());
    module->mOutputFromElse.resize(output->size());
    for (int i=0; i<output->size(); ++i) {
        auto data = output->GetAs<StringVec>(i);
        MNN_ASSERT(data->data()->size() == 2);

        auto thenPos = _findPos(thenG.outputs, data->data()->GetAsString(0)->str());
        MNN_ASSERT(thenPos >= 0);
        auto elsePos = _findPos(elseG.outputs, data->data()->GetAsString(1)->str());
        module->mOutputFromThen[i] = thenPos;
        module->mOutputFromElse[i] = elsePos;
    }
    return module;
}

Module* IfModule::clone(CloneContext* ctx) const {
    IfModule* module(new IfModule);
    module->mInputForThen = mInputForThen;
    module->mInputForElse = mInputForElse;
    module->mOutputFromThen = mOutputFromThen;
    module->mOutputFromElse = mOutputFromElse;
    module->mThen.reset(mThen->clone(ctx));
    module->mElse.reset(mElse->clone(ctx));
    return this->cloneBaseTo(ctx, module);
}

}  // namespace Express
}  // namespace MNN
