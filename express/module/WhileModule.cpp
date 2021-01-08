//
//  WhileModule.cpp
//  MNN
//
//  Created by MNN on b'2020/09/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "WhileModule.hpp"
#include "StaticModule.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include "MNN_generated.h"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
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
WhileModule* WhileModule::create(const Op* op, const std::map<std::string, SubGraph>& subGraph) {
    auto module = new WhileModule;
    module->setType("WhileModule");
    auto whileParam = op->main_as_WhileParam();
    auto& body = subGraph.find(whileParam->body_graph()->str())->second;
    auto& cond = subGraph.find(whileParam->cond_graph()->str())->second;
    module->mBody = body.m;
    module->mCond = cond.m;
    /** Compute map index
     int mCondInputNumber;
     int mBodyInputNumber;
     
     // First mCondInputs' index, Second: inputs's index
     std::vector<std::pair<int, int>> mInputForCond;

     // First mBodyInputs' index, Second: inputs's index
     std::vector<std::pair<int, int>> mInputForBody;
     std::vector<int> mOutputFromBody;
     std::vector<std::pair<int, int>> mUpdateForCond;
     std::vector<std::pair<int, int>> mUpdateForBody;
     std::vector<std::pair<int, int>> mCondUpdateForCond;
     std::vector<std::pair<int, int>> mCondUpdateForBody;
     */
    // Map Inputs
    module->mBodyInputNumber = body.inputs.size();
    module->mCondInputNumber = cond.inputs.size();
    for (int i=0; i<whileParam->aliases_inputs()->size(); ++i) {
        auto index = i;
        auto data = whileParam->aliases_inputs()->GetAs<StringVec>(i);
        for (int s=0; s<data->data()->size(); ++s) {
            auto name = data->data()->GetAsString(s)->str();
            auto bodyInputPos = _findPos(body.inputs, name);
            if (bodyInputPos >= 0) {
                module->mInputForBody.emplace_back(std::make_pair(bodyInputPos, i));
            }
            auto condInputPos = _findPos(cond.inputs, name);
            if (condInputPos >= 0) {
                module->mInputForCond.emplace_back(std::make_pair(condInputPos, i));
            }
        }
    }
    // Map update
    auto update = whileParam->aliases_updates();
    std::set<int> reusedTensors;
    std::map<int, int> replaceOutputs;
    for (int i=0; i<update->size(); ++i) {
        auto data = update->GetAs<StringVec>(i);
        int bodyInputPos = -1;
        int condInputPos = -1;
        int bodyOutputPos = -1;
        int condOutputPos = -1;
        MNN_ASSERT(2 == data->data()->size());
        auto outputName = data->data()->GetAsString(0)->str();
        auto inputName = data->data()->GetAsString(1)->str();
        bodyInputPos = _findPos(body.inputs, inputName);
        condInputPos = _findPos(cond.inputs, inputName);
        bodyOutputPos = _findPos(body.outputs, outputName);
        condOutputPos = _findPos(cond.outputs, outputName);

        auto updateBodyOutputPos = _findPos(body.outputs, inputName);
        
        MNN_ASSERT(bodyOutputPos == -1 || condOutputPos == -1);
        if (condOutputPos >= 0) {
            if (bodyInputPos >= 0) {
                module->mCondUpdateForBody.emplace_back(std::make_pair(bodyInputPos, condOutputPos));
            }
            if (condInputPos >= 0) {
                module->mCondUpdateForCond.emplace_back(std::make_pair(condInputPos, condOutputPos));
            }
        }
        if (bodyOutputPos >= 0) {
            if (bodyInputPos >= 0) {
                reusedTensors.insert(bodyOutputPos);
                module->mUpdateForBody.emplace_back(std::make_pair(bodyInputPos, bodyOutputPos));
            }
            if (condInputPos >= 0) {
                module->mUpdateForCond.emplace_back(std::make_pair(condInputPos, bodyOutputPos));
            }
            if (updateBodyOutputPos >= 0) {
                replaceOutputs.insert(std::make_pair(updateBodyOutputPos, bodyOutputPos));
            }
        }
    }
    // Map outputs
    auto output = whileParam->aliases_outputs();
    for (int i=0; i<output->size(); ++i) {
        auto data = output->GetAsString(i);
        auto pos = _findPos(body.outputs, data->str());
        MNN_ASSERT(pos >= 0);
        if (replaceOutputs.find(pos) != replaceOutputs.end()) {
            pos = replaceOutputs[pos];
        }
        module->mOutputFromBody.emplace_back(pos);
    }
    if (module->mBody->type() == "StaticModule") {
        static_cast<StaticModule*>(module->mBody.get())->setReusedTensors(reusedTensors);
    }
    return module;
}

std::vector<Express::VARP> WhileModule::onForward(const std::vector<Express::VARP>& inputsI) {
    std::vector<Express::VARP> condInputs(mCondInputNumber);
    std::vector<Express::VARP> bodyInputs(mBodyInputNumber);
    auto& inputs = inputsI;
    for (auto& p : mInputForCond) {
        condInputs[p.first] = inputs[p.second];
    }
    for (auto& p : mInputForBody) {
        bodyInputs[p.first] = inputs[p.second];
    }

    std::vector<Express::VARP> outputs(mOutputFromBody.size());
    while (true) {
        auto res = mCond->onForward(condInputs)[0];
        auto resPtr = res->readMap<int>();
        if (resPtr[0] <= 0) {
            break;
        }
        auto bodyOutputs = mBody->onForward(bodyInputs);
        Express::Variable::prepareCompute(bodyOutputs);
        for (int i=0; i<bodyOutputs.size(); ++i) {
            auto p = bodyOutputs[i];
            if (p->expr().first->get() != nullptr) {
                auto ptr = p->readMap<void>();
                auto info = p->getInfo();
                auto newV = Express::_Input(info->dim, info->order, info->type);
                if (nullptr != ptr) {
                    ::memcpy(newV->writeMap<void>(), ptr, info->type.bytes() * info->size);
                }
                bodyOutputs[i] = newV;
            }
        }
        for (int i=0; i<mOutputFromBody.size(); ++i) {
            outputs[i] = bodyOutputs[mOutputFromBody[i]];
        }
        for (auto& p : mUpdateForCond) {
            condInputs[p.first] = bodyOutputs[p.second];
        }
        for (auto& p : mUpdateForBody) {
            bodyInputs[p.first] = bodyOutputs[p.second];
        }
        for (auto& p : mCondUpdateForCond) {
            condInputs[p.first] = res;
        }
        for (auto& p : mCondUpdateForBody) {
            bodyInputs[p.first] = res;
        }
    }
    return outputs;
}

Module* WhileModule::clone(CloneContext* ctx) const {
    WhileModule* module(new WhileModule);
    module->mCondInputNumber = mCondInputNumber;
    module->mBodyInputNumber = mBodyInputNumber;
    module->mInputForCond = mInputForCond;
    module->mInputForBody = mInputForBody;
    module->mOutputFromBody = mOutputFromBody;
    module->mUpdateForCond = mUpdateForCond;
    module->mUpdateForBody = mUpdateForBody;
    module->mCondUpdateForCond = mCondUpdateForCond;
    module->mCondUpdateForBody = mCondUpdateForBody;
    module->mCond.reset(mCond->clone(ctx));
    module->mBody.reset(mBody->clone(ctx));
    return this->cloneBaseTo(ctx, module);
}

};
};
