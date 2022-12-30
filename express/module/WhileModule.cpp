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
#include <MNN/expr/ExecutorScope.hpp>
#include "RuntimeAttr.hpp"
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
    std::shared_ptr<WhileModule::Info> info(new WhileModule::Info);
    module->mInfo = info;
    if (nullptr != op->name()) {
        module->setName(op->name()->str());
    }
    auto whileParam = op->main_as_WhileParam();
    auto& body = subGraph.find(whileParam->body_graph()->str())->second;
    module->mBody = body.m;
    module->registerModel({body.m});
    if (whileParam->cond_graph() == nullptr) {
        // From onnx's loop, use easy way to init
        info->mOutputNumber = op->outputIndexes()->size();
        info->mBodyInputNumber = op->inputIndexes()->size();
        return module;
    }
    auto& cond = subGraph.find(whileParam->cond_graph()->str())->second;
    module->mCond = cond.m;
    module->registerModel({cond.m});
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
    info->mBodyInputNumber = body.inputs.size();
    info->mCondInputNumber = cond.inputs.size();
    for (int i=0; i<whileParam->aliases_inputs()->size(); ++i) {
        auto index = i;
        auto data = whileParam->aliases_inputs()->GetAs<StringVec>(i);
        for (int s=0; s<data->data()->size(); ++s) {
            auto name = data->data()->GetAsString(s)->str();
            auto bodyInputPos = _findPos(body.inputs, name);
            if (bodyInputPos >= 0) {
                info->mInputForBody.emplace_back(std::make_pair(bodyInputPos, i));
            }
            auto condInputPos = _findPos(cond.inputs, name);
            if (condInputPos >= 0) {
                info->mInputForCond.emplace_back(std::make_pair(condInputPos, i));
            }
//            if (bodyInputPos < 0 && condInputPos < 0) {
//                MNN_ASSERT(false);
//            }
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
                info->mCondUpdateForBody.emplace_back(std::make_pair(bodyInputPos, condOutputPos));
            }
            if (condInputPos >= 0) {
                info->mCondUpdateForCond.emplace_back(std::make_pair(condInputPos, condOutputPos));
            }
        }
        if (bodyOutputPos >= 0) {
            if (bodyInputPos >= 0) {
                reusedTensors.insert(bodyOutputPos);
                info->mUpdateForBody.emplace_back(std::make_pair(bodyInputPos, bodyOutputPos));
            }
            if (condInputPos >= 0) {
                info->mUpdateForCond.emplace_back(std::make_pair(condInputPos, bodyOutputPos));
            }
            if (updateBodyOutputPos >= 0) {
                replaceOutputs.insert(std::make_pair(updateBodyOutputPos, bodyOutputPos));
            }
            MNN_ASSERT(condInputPos >= 0 || bodyInputPos >= 0);
        }
        //MNN_ASSERT(bodyOutputPos >= 0 || condOutputPos >= 0);
    }
    MNN_ASSERT(!info->mUpdateForCond.empty());
    // Map outputs
    auto output = whileParam->aliases_outputs();
    info->mOutputNumber = output->size();
    for (int i=0; i<output->size(); ++i) {
        auto data = output->GetAsString(i);
        auto pos = _findPos(body.outputs, data->str());
        auto posInput = _findPos(body.inputs, data->str());
        //MNN_ASSERT(pos >= 0 || posInput >= 0);
        if (replaceOutputs.find(pos) != replaceOutputs.end()) {
            pos = replaceOutputs[pos];
        }
        if (pos >= 0) {
            info->mOutputFromBody.emplace_back(std::make_pair(i, pos));
        }
        if (posInput >= 0) {
            info->mOutputFromBodyInput.emplace_back(std::make_pair(i, posInput));
        }
        for (int j=0; j<whileParam->aliases_inputs()->size(); ++j) {
            auto inputStrVec = whileParam->aliases_inputs()->GetAs<StringVec>(j);
            bool find = false;
            for (int k=0; k<inputStrVec->data()->size(); ++k) {
                auto name = inputStrVec->data()->GetAsString(k)->str();
                if (name == data->str()) {
                    find = true;
                    info->mOutputFromInput.emplace_back(j);
                    break;
                }
            }
            if (find) {
                break;
            }
        }
    }
    return module;
}

std::vector<Express::VARP> WhileModule::onForward(const std::vector<Express::VARP>& inputsI) {
    std::vector<Express::VARP> bodyInputs(mInfo->mBodyInputNumber);
    auto& inputs = inputsI;
    int step = 0;
    std::vector<Express::VARP> outputs(mInfo->mOutputNumber);
    if (mCond == nullptr) {
        auto limit = inputs[0]->readMap<int>()[0];
        int cond = inputs[1]->readMap<int>()[0];
        // Body Input: 2 + N, Body Output: 1 + N + K, Op output: N + K
        int N = mInfo->mBodyInputNumber - 2;
        int K = mInfo->mOutputNumber - N;
        std::vector<std::vector<VARP>> spans(K);
        std::vector<VARP> bodyOutputs;
        for (int i=0; i<N; ++i) {
            outputs[i] = inputs[i+2];
        }
        if (limit > 0 && cond > 0) {
            bodyInputs = inputs;
            bodyInputs[0] = _Input({}, NCHW, halide_type_of<int>());
            while (step < limit && cond > 0) {
                bodyInputs[0]->writeMap<int>()[0] = step;
                bodyOutputs = mBody->onForward(bodyInputs);
                if (bodyOutputs.empty()) {
                    // Has Error
                    return {};
                }
                for (int i=0; i<N; ++i) {
                    bodyInputs[i + 2] = bodyOutputs[i + 1];
                }
                for (int i=0; i<K; ++i) {
                    spans[i].emplace_back(bodyOutputs[1+N+i]);
                }
                step++;
                cond = bodyOutputs[0]->readMap<int>()[0];
            }
            for (int i=0; i<N; ++i) {
                outputs[i] = bodyOutputs[i+1];
            }
        }
        for (int i=0; i<K; ++i) {
            outputs[i+N] = _Stack(spans[i]);
        }
        return outputs;
    }
    std::vector<Express::VARP> condInputs(mInfo->mCondInputNumber);
    for (auto& p : mInfo->mInputForCond) {
        condInputs[p.first] = inputs[p.second];
    }
    for (auto& p : mInfo->mInputForBody) {
        bodyInputs[p.first] = inputs[p.second];
    }

    for (int i = 0; i < mInfo->mOutputFromInput.size(); ++i) {
        outputs[i] = inputs[mInfo->mOutputFromInput[i]];
    }
    while (true) {
        VARP res;
        {
            auto condOutputs = mCond->onForward(condInputs);
            if (condOutputs.empty()) {
                return {};
            }
            res = condOutputs[0];
        }
        auto resPtr = res->readMap<int>();
        if (resPtr[0] <= 0) {
            break;
        }
        step++;
        // MNN_PRINT("before while op name: %s, step:%d\n", name().c_str(), step);
        auto bodyOutputs = mBody->onForward(bodyInputs);
        if (bodyOutputs.empty()) {
            return {};
        }
        for (auto& p : mInfo->mUpdateForCond) {
            condInputs[p.first] = bodyOutputs[p.second];
        }
        for (auto& p : mInfo->mUpdateForBody) {
            bodyInputs[p.first] = bodyOutputs[p.second];
        }
        for (auto& p : mInfo->mCondUpdateForCond) {
            condInputs[p.first] = res;
        }
        for (auto& p : mInfo->mCondUpdateForBody) {
            bodyInputs[p.first] = res;
        }
        for (int i=0; i<mInfo->mOutputFromBody.size(); ++i) {
            outputs[mInfo->mOutputFromBody[i].first] = bodyOutputs[mInfo->mOutputFromBody[i].second];
        }
        for (int i=0; i<mInfo->mOutputFromBodyInput.size(); ++i) {
            outputs[mInfo->mOutputFromBodyInput[i].first] = bodyInputs[mInfo->mOutputFromBodyInput[i].second];
        }
    }
    for (auto o : outputs) {
        MNN_ASSERT(nullptr != o);
    }
    return outputs;
}

Module* WhileModule::clone(CloneContext* ctx) const {
    WhileModule* module(new WhileModule);
    module->mInfo = mInfo;
    if (nullptr != mCond.get()) {
        module->mCond.reset(mCond->clone(ctx));
        module->registerModel({module->mCond});
    }
    module->mBody.reset(mBody->clone(ctx));
    module->registerModel({module->mBody});
    return this->cloneBaseTo(ctx, module);
}

};
};
