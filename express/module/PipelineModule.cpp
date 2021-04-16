//
//  PipelineModule.cpp
//  MNN
//
//  Created by MNN on 2020/01/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "PipelineModule.hpp"
#include "MNN_generated.h"
#include <set>
#include <vector>
#include "StaticModule.hpp"
#include "IfModule.hpp"
#include "WhileModule.hpp"
#include "Utils.hpp"
#include "core/Backend.hpp"
#include <MNN/expr/ExecutorScope.hpp>
using namespace MNN::Express;
namespace MNN {
namespace Express {
//#define DYNAMIC
class ExprModule : public Module {
public:
    ExprModule(EXPRP expr) {
        mExpr   = expr;
        setName(expr->name());
        mInputs = expr->inputs();
        auto op = mExpr->get();
        if (op) {
            auto typeName = EnumNameOpType(op->type());
            setType(typeName);
        }
        for (int i = 0; i < mInputs.size(); ++i) {
            auto inputExpr = mInputs[i]->expr().first;
            if (inputExpr->get() != nullptr) {
                mInputs[i] = nullptr;
                mInputIndexes.emplace_back(i);
                continue;
            }
            switch (inputExpr->inputType()) {
                case VARP::INPUT:
                    mInputs[i] = nullptr;
                    mInputIndexes.emplace_back(i);
                    break;
                case VARP::CONSTANT:
                    break;
                case VARP::TRAINABLE:
                    addParameter(mInputs[i]);
                    break;
                default:
                    break;
            }
        }
    }
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override {
        MNN_ASSERT(mInputIndexes.size() == inputs.size());
        if (nullptr == mExpr->get()) {
            return {Variable::create(mExpr)};
        }
        std::vector<VARP> tempInputs = mInputs;
        for (int i = 0; i < inputs.size(); ++i) {
            tempInputs[mInputIndexes[i]] = inputs[i];
        }
        std::vector<VARP> outputVars;
        auto newExpr = Expr::create(mExpr->extra(), std::move(tempInputs), mExpr->outputSize());
        newExpr->setName(mExpr->name());
        for (int i = 0; i < mExpr->outputSize(); ++i) {
            outputVars.emplace_back(Variable::create(newExpr, i));
        }
        return outputVars;
    }
    const std::vector<int>& inputIndexes() const {
        return mInputIndexes;
    }

private:
    Module* clone(CloneContext* ctx) const override {
        ExprModule* module(new ExprModule(ctx->getOrClone(mExpr)));
        for (const VARP& var : mInputs) {
            module->mInputs.push_back(ctx->getOrClone(var));
        }
        module->mInputIndexes = mInputIndexes;
        return this->cloneBaseTo(ctx, module);
    }

    EXPRP mExpr;
    std::vector<VARP> mInputs;
    std::vector<int> mInputIndexes;
};

PipelineModule::PipelineModule(std::vector<VARP> inputs, std::vector<VARP> outputs, const Transformer& transformFunction) {
    setType(PIPELINE_MODULE);
    std::vector<EXPRP> executeOrder;
    std::set<EXPRP> inputExpr;
    for (auto v : inputs) {
        inputExpr.insert(v->expr().first);
    }
    for (auto output : outputs) {
        Expr::visit(output->expr().first,
        [&executeOrder, &inputExpr](EXPRP expr) {
            if (expr->visited()) {
                return false;
            }
            if (inputExpr.find(expr)!= inputExpr.end()) {
                expr->setVisited(true);
                executeOrder.emplace_back(expr);
                return false;
            }
            return true;
        },
        [&executeOrder](EXPRP expr) {
            //FUNC_PRINT_ALL(var->name().c_str(), s);
            if (!expr->visited()) {
                executeOrder.emplace_back(expr);
                expr->setVisited(true);
            }
            return true;
        });
    }
    for (auto expr : executeOrder) {
        expr->setVisited(false);
    }
    // Set Indexes
    std::map<EXPRP, int> indexes;
    int currentIndexes = 0;
    for (auto expr : executeOrder) {
        indexes[expr] = currentIndexes;
        currentIndexes += expr->outputSize();
    }
    std::set<EXPRP> inputSets;
    mInputIndexes.clear();
    mStackSize = currentIndexes;
    for (auto v : inputs) {
        auto inputExpr = v->expr();
        mInputIndexes.emplace_back(indexes[inputExpr.first] + inputExpr.second);
        inputSets.insert(inputExpr.first);
    }

    // Create All SubModule
    for (auto expr : executeOrder) {
        if (inputSets.find(expr) != inputSets.end()) {
            continue;
        }
        std::pair<std::vector<int>, std::shared_ptr<Module> > moduleResult;
        bool extracted = false;
        if (!transformFunction) {
            moduleResult = std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
        } else {
            moduleResult = transformFunction(expr);
        }
        if (moduleResult.second == nullptr) {
            std::shared_ptr<Module> module(new ExprModule(expr));
            moduleResult.first  = ((ExprModule*)module.get())->inputIndexes();
            moduleResult.second = module;
        } else {
            extracted = true;
        }
        auto subInputs        = expr->inputs();
        auto& exprInputIndexes = moduleResult.first;
        std::vector<int> inputIndexes;
        if (exprInputIndexes.empty() && extracted) {
            inputIndexes.resize(subInputs.size());
            for (int i = 0; i < inputIndexes.size(); ++i) {
                auto inputExpr  = subInputs[i]->expr();
                inputIndexes[i] = indexes[inputExpr.first] + inputExpr.second;
            }
        } else {
            inputIndexes.resize(exprInputIndexes.size());
            for (int i = 0; i < inputIndexes.size(); ++i) {
                auto inputExpr  = subInputs[exprInputIndexes[i]]->expr();
                inputIndexes[i] = indexes[inputExpr.first] + inputExpr.second;
            }
        }
        std::vector<int> outputIndexes(expr->outputSize());
        for (int i = 0; i < outputIndexes.size(); ++i) {
            outputIndexes[i] = indexes[expr] + i;
        }
        mSubModules.emplace_back(std::make_tuple(moduleResult.second, inputIndexes, outputIndexes));
        registerModel({moduleResult.second});
    }
    mOutputIndexes.clear();
    for (auto output : outputs) {
        auto outputExpr = output->expr();
        mOutputIndexes.emplace_back(indexes[outputExpr.first] + outputExpr.second);
    }
}

std::vector<int> PipelineModule::countOutputReference(std::vector<int> outputIndices) {
    MNN_ASSERT(outputIndices.size() > 0);
    std::vector<int> countResult(outputIndices.size(), 0);

    for (int i = 0; i < mSubModules.size(); i++) {
        auto &m = mSubModules[i];
        auto& theModule = std::get<0>(m);
        auto name = theModule->name();
        auto &inputIndices = std::get<1>(m);

        for (int j = 0; j < inputIndices.size(); j++) {
            int index = inputIndices[j];
            for (int k = 0; k < countResult.size(); k++) {
                if (index == outputIndices[k]) {
                    countResult[k]++;
                }
            }
        }
    }
    return countResult;
}

std::vector<VARP> PipelineModule::onForward(const std::vector<VARP>& inputs) {
    std::vector<VARP> mStack(mStackSize);
    for (int i = 0; i < mInputIndexes.size(); ++i) {
        mStack[mInputIndexes[i]] = inputs[i];
    }
    for (int index = 0; index < mSubModules.size(); ++index) {
        auto& m = mSubModules[index];
        std::vector<VARP> tempInputs(std::get<1>(m).size());
        for (int i = 0; i < tempInputs.size(); ++i) {
            tempInputs[i] = mStack[std::get<1>(m)[i]];
            MNN_ASSERT(nullptr != tempInputs[i]);
        }
        std::vector<VARP> tempOutputs = std::get<0>(m)->onForward(tempInputs);
        MNN_ASSERT(tempOutputs.size() == std::get<2>(m).size());
        for (int i = 0; i < tempOutputs.size(); ++i) {
            mStack[std::get<2>(m)[i]] = tempOutputs[i];
            MNN_ASSERT(nullptr != tempOutputs[i]);
        }
    }
    std::vector<VARP> outputs(mOutputIndexes.size());
    for (int i = 0; i < mOutputIndexes.size(); ++i) {
        outputs[i] = mStack[mOutputIndexes[i]];
    }
    return outputs;
}
void PipelineModule::onClearCache() {
    // Do nothing
}

void PipelineModule::_createSubGraph(const MNN::Net* net, const Module::Config* config, std::map<std::string, SubGraph>& subGraphMap) {
    auto subGraphs = net->subgraphs();
    if (nullptr == subGraphs) {
        return;
    }
    for (int i=0; i<subGraphs->size(); ++i) {
        auto graph = subGraphs->GetAs<SubGraphProto>(i);
        std::vector<std::string> subInputs;
        std::vector<std::string> subOutputs;
        if (nullptr != graph->inputs()) {
            for (int v=0; v<graph->inputs()->size(); ++v) {
                auto index = graph->inputs()->data()[v];
                subInputs.emplace_back(graph->tensors()->GetAsString(index)->str());
            }
        }
        for (int v=0; v<graph->outputs()->size(); ++v) {
            auto index = graph->outputs()->data()[v];
            subOutputs.emplace_back(graph->tensors()->GetAsString(index)->str());
        }
        // Pack to Net for loading
        std::shared_ptr<Module> submodule;
        {
            std::unique_ptr<SubGraphProtoT> _tempInfo(graph->UnPack());
            std::unique_ptr<NetT> _tempNet(new NetT);
            _tempNet->oplists = std::move(_tempInfo->nodes);
            _tempNet->tensorName = std::move(_tempInfo->tensors);
            _tempNet->extraTensorDescribe = std::move(_tempInfo->extraTensorDescribe);
            flatbuffers::FlatBufferBuilder builder(1024);
            auto offset = Net::Pack(builder, _tempNet.get());
            builder.Finish(offset);
            submodule.reset(PipelineModule::load(subInputs, subOutputs, (const uint8_t*)builder.GetBufferPointer(), builder.GetSize(), config, subGraphMap, true));
            if (graph->name() != nullptr) {
                submodule->setName(graph->name()->str());
            }
        }
        auto key = graph->name()->str();
        SubGraph subgraph;
        subgraph.inputs = std::move(subInputs);
        subgraph.outputs = std::move(subOutputs);
        subgraph.m = submodule;
        subGraphMap.insert(std::make_pair(key, subgraph));
    }
    return;
}

struct SubModuleInfo {
    std::vector<int> opList;
    std::vector<int> inputs;;
    std::vector<int> outputs;
    std::vector<uint8_t> tensorMask;
};
static std::vector<SubModuleInfo> _createSubModuleInfo(const MNN::Net* net, const std::set<int>& inputIndexes, const std::set<int>& outputIndexes) {
    std::vector<SubModuleInfo> submodule;
    SubModuleInfo current;
    std::vector<int> inputOps;

    // Seperate the graph to serveral submodule
    for (int i=0; i<net->oplists()->size(); ++i) {
        auto op = net->oplists()->GetAs<Op>(i);
        // Collect Input
        if (op->type() == OpType_Input) {
            inputOps.emplace_back(i);
            continue;
        }
        if (op->type() == OpType_If || op->type() == OpType_While || op->type() == OpType_Where) {
            if (current.opList.size() > 0) {
                // Not empty
                submodule.emplace_back(std::move(current));
            }
            SubModuleInfo controlOp;
            controlOp.opList = {i};
            submodule.emplace_back(std::move(controlOp));
            continue;
        }
        current.opList.emplace_back(i);
    }
    if (!current.opList.empty()) {
        submodule.emplace_back(std::move(current));
    }

    /**Compute All SubModule's inputs and outputs*/
    // 0: not use, 1: input, 2: output, 3: mid, 4: valid output
    for (int moduleIndex=0; moduleIndex < submodule.size(); ++moduleIndex) {
        auto& m = submodule[moduleIndex];
        if (1 == m.opList.size()) {
            // Fast way to determine
            auto op = net->oplists()->GetAs<Op>(m.opList[0]);
            if (nullptr != op->inputIndexes()) {
                m.inputs.resize(op->inputIndexes()->size());
                ::memcpy(m.inputs.data(), op->inputIndexes()->data(), m.inputs.size() * sizeof(int));
            }
            if (nullptr != op->outputIndexes()) {
                m.outputs.resize(op->outputIndexes()->size());
                ::memcpy(m.outputs.data(), op->outputIndexes()->data(), m.outputs.size() * sizeof(int));
            }
        } else {
            m.tensorMask = std::vector<uint8_t>(net->tensorName()->size(), 0);
            auto& tensorMask = m.tensorMask;
            for (auto opIndex : m.opList) {
                auto op = net->oplists()->GetAs<Op>(opIndex);
                if (nullptr != op->inputIndexes()) {
                    for (int v=0; v<op->inputIndexes()->size(); ++v) {
                        auto index = op->inputIndexes()->data()[v];
                        tensorMask[index] = tensorMask[index] | 1;
                    }
                }
                if (nullptr != op->outputIndexes()) {
                    for (int v=0; v<op->outputIndexes()->size(); ++v) {
                        auto index = op->outputIndexes()->data()[v];
                        tensorMask[index] = tensorMask[index] | 2;
                    }
                }
            }
            for (int i=0; i<tensorMask.size(); ++i) {
                if (0 == tensorMask[i]) {
                    continue;
                }
                if (1 == tensorMask[i]) {
                    m.inputs.emplace_back(i);
                    continue;
                }
                if (2 == tensorMask[i]) {
                    m.outputs.emplace_back(i);
                    continue;
                }
                if (3 == tensorMask[i]) {
                    if (outputIndexes.find(i) != outputIndexes.end()) {
                        m.outputs.emplace_back(i);
                    }
                }
            }
        }
        // Check if the module's input is valid
        for (int i=0; i<m.inputs.size(); ++i) {
            auto index = m.inputs[i];
            if (inputIndexes.find(index) != inputIndexes.end()) {
                continue;
            }
            bool find = false;
            for (int sub=0; sub < moduleIndex; ++sub) {
                for (auto out : submodule[sub].outputs) {
                    if (out == index) {
                        find = true;
                        break;
                    }
                }
                if (find) {
                    break;
                }
            }
            if (find) {
                continue;
            }
            // Find from module
            for (int sub=0; sub < moduleIndex; ++sub) {
                if (submodule[sub].tensorMask.empty()) {
                    continue;
                }
                if (submodule[sub].tensorMask[index] == 2) {
                    find = true;
                    break;
                }
                if (submodule[sub].tensorMask[index] == 3) {
                    submodule[sub].outputs.emplace_back(index);
                    submodule[sub].tensorMask[index] = 2;
                    find = true;
                    break;
                }
            }
            if (!find) {
                if (net->tensorName() != nullptr) {
                    MNN_PRINT("%d tensor [ %s ] is input but not found\n", index, net->tensorName()->GetAsString(index)->c_str());
                }
            }
            MNN_ASSERT(find);
        }
    }
    for (auto& m : submodule) {
        m.tensorMask.clear();
    }
    return submodule;
}

static Module* _createSubModule(const MNN::Net* net, const SubModuleInfo& info, const std::map<std::string, SubGraph>& subs, const Module::Config& config, bool inRecurse) {
    if (1 == info.opList.size()) {
        auto op = net->oplists()->GetAs<Op>(info.opList[0]);
        if (OpType_If == op->type()) {
            return IfModule::create(op, subs);
        }
        if (OpType_While == op->type()) {
            return WhileModule::create(op, subs);
        }
        // MNN_ASSERT(false);
    }
    std::unique_ptr<NetT> _tempNet(new NetT);
    // Copy Tensor Name
    _tempNet->tensorName.resize(net->tensorName()->size());
    for (int i=0; i<net->tensorName()->size(); ++i) {
        _tempNet->tensorName[i] = net->tensorName()->GetAsString(i)->str();
    }
    // Copy Tensor Describe for quant model
    if (net->extraTensorDescribe()) {
        _tempNet->extraTensorDescribe.resize(net->extraTensorDescribe()->size());
        for (int i=0; i<net->extraTensorDescribe()->size(); ++i) {
            _tempNet->extraTensorDescribe[i].reset(net->extraTensorDescribe()->Get(i)->UnPack());
        }
    }
    // Create Input node
    std::vector<std::string> inputNames;
    for (auto index : info.inputs) {
        std::unique_ptr<OpT> inputOp(new OpT);
        inputOp->outputIndexes = {index};
        inputOp->type = OpType_Input;
        inputOp->main.type = OpParameter_Input;
        inputOp->main.value = new InputT;
        inputOp->main.AsInput()->dims = {0, 0, -1, -1};
        _tempNet->oplists.emplace_back(std::move(inputOp));
        inputNames.emplace_back(_tempNet->tensorName[index]);
    }
    // Create compute node
    for (auto opIndex : info.opList) {
        std::unique_ptr<OpT> op(net->oplists()->GetAs<Op>(opIndex)->UnPack());
        _tempNet->oplists.emplace_back(std::move(op));
    }
    // Get output names
    std::vector<std::string> outputNames;
    for (auto index : info.outputs) {
        outputNames.emplace_back(_tempNet->tensorName[index]);
    }
    // Create Net Buffer
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = Net::Pack(builder, _tempNet.get());
    builder.Finish(offset);
    _tempNet.reset();
    return new StaticModule((const uint8_t*)builder.GetBufferPointer(), builder.GetSize(), inputNames, outputNames, config, inRecurse);
}

Module* PipelineModule::load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, const Module::Config* config) {
    // Create Subgraph
    auto net = GetNet(buffer);
    Module::Config defaultConfig;
    if (nullptr == config) {
        config = &defaultConfig;
    }
    auto subGraphs = net->subgraphs();
    if (nullptr == net->oplists() || nullptr == net->tensorName()) {
        MNN_ERROR("Invalid net, for null oplist or tensorName\n");
        return nullptr;
    }
    std::map<std::string, SubGraph> subGraphMap;
    _createSubGraph(net, config, subGraphMap);
    return load(inputs, outputs, buffer, length, config, subGraphMap);
}
Module* PipelineModule::load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, const Module::Config* config, std::map<std::string, SubGraph>& subGraphMap, bool inRecurce) {
    auto net = GetNet(buffer);
    if (!config->dynamic) {
        bool linear = true;
        auto iter = net->oplists()->begin();
        for (; iter != net->oplists()->end(); iter++) {
            if (iter->type() == OpType_While) {
                linear = false;
                break;
            }
            if (iter->type() == OpType_If) {
                linear = false;
                break;
            }
            if (iter->type() == OpType_Where) {
                linear = false;
                break;
            }
        }
        if (linear) {
            // Has no control flow and WhereOp, can just use static module
            return new StaticModule(buffer, length, inputs, outputs, *config, false);
        }
    }
    std::set<int> inputIndexes;
    std::set<int> outputIndexes;
    std::map<std::string, int> inputsMap;
    std::map<std::string, int> outputsMap;
    for (int i=0; i<net->tensorName()->size(); ++i) {
        auto tname = net->tensorName()->GetAsString(i)->str();
        for (auto& s : inputs) {
            if (tname == s) {
                inputIndexes.emplace(i);
                inputsMap.insert(std::make_pair(s, i));
                break;
            }
        }
        for (auto& s : outputs) {
            if (tname == s) {
                outputIndexes.emplace(i);
                outputsMap.insert(std::make_pair(s, i));
                break;
            }
        }
    }
    std::vector<int> inputIndexesVec(inputs.size());
    for (int i=0; i<inputs.size(); ++i) {
        inputIndexesVec[i] = inputsMap[inputs[i]];
    }
    std::vector<int> outputIndexesVec(outputs.size());
    for (int i=0; i<outputs.size(); ++i) {
        outputIndexesVec[i] = outputsMap[outputs[i]];
    }

    auto subModulesInfo = _createSubModuleInfo(net, inputIndexes, outputIndexes);
    std::vector<std::shared_ptr<Module>> subModules(subModulesInfo.size());
    for (int i=0; i<subModulesInfo.size(); ++i) {
        subModules[i].reset(_createSubModule(net, subModulesInfo[i], subGraphMap, *config, inRecurce));
    }
    auto result = new PipelineModule;
    /**
     Compute:
     std::vector<std::tuple<std::shared_ptr<Module>, std::vector<int>, std::vector<int>>> mSubModules;
     std::vector<int> mInputIndexes;
     std::vector<int> mOutputIndexes;
     int mStackSize = 0;
     */
    // Make Stack, first: origin, second: new
    std::map<int, int> stackMap;
    int stackIndex = 0;
    for (auto index : inputIndexesVec) {
        if (stackMap.find(index) == stackMap.end()) {
            stackMap.insert(std::make_pair(index, stackIndex));
            stackIndex++;
        }
    }
    for (auto& m : subModulesInfo) {
        for (auto index : m.inputs) {
            if (stackMap.find(index) == stackMap.end()) {
                stackMap.insert(std::make_pair(index, stackIndex));
                stackIndex++;
            }
        }
        for (auto index : m.outputs) {
            if (stackMap.find(index) == stackMap.end()) {
                stackMap.insert(std::make_pair(index, stackIndex));
                stackIndex++;
            }
        }
    }
    result->mStackSize = stackMap.size();
    MNN_ASSERT(result->mStackSize > 0);
    for (int i=0; i<subModulesInfo.size(); ++i) {
        auto& info = subModulesInfo[i];
        // Reindex stack index
        std::vector<int> subInputs(info.inputs.size());
        for (int i=0; i<info.inputs.size(); ++i) {
            subInputs[i] = stackMap[info.inputs[i]];
        }
        std::vector<int> subOutputs(info.outputs.size());
        for (int i=0; i<info.outputs.size(); ++i) {
            subOutputs[i] = stackMap[info.outputs[i]];
        }
        result->mSubModules.emplace_back(std::make_tuple(subModules[i], subInputs, subOutputs));
    }
    for (int i=0; i<inputIndexesVec.size(); ++i) {
        inputIndexesVec[i] = stackMap[inputIndexesVec[i]];
    }
    for (int i=0; i<outputIndexesVec.size(); ++i) {
        outputIndexesVec[i] = stackMap[outputIndexesVec[i]];
    }
    result->mInputIndexes = std::move(inputIndexesVec);
    result->mOutputIndexes = std::move(outputIndexesVec);

    return result;

}

Module* PipelineModule::clone(CloneContext* ctx) const {
    PipelineModule* module(new PipelineModule);
    for (const auto& it : mSubModules) {
        const std::shared_ptr<Module>& submodule = std::get<0>(it);
        const std::vector<int>& input_indices = std::get<1>(it);
        const std::vector<int>& output_indices = std::get<2>(it);
        std::shared_ptr<Module> replica_submodule(submodule->clone(ctx));
        module->mSubModules.push_back(
            std::make_tuple(replica_submodule, input_indices, output_indices));
        module->registerModel({replica_submodule});
    }
    module->mInputIndexes = mInputIndexes;
    module->mOutputIndexes = mOutputIndexes;
    module->mStackSize = mStackSize;
    return this->cloneBaseTo(ctx, module);
}


} // namespace Express
} // namespace MNN
