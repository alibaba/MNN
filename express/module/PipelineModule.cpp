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
#define PIPELINE_MODULE "_pipeline_module__"
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

Module* PipelineModule::extract(std::vector<Express::VARP> inputs, std::vector<Express::VARP> outputs, bool fortrain, const std::map<std::string, SubGraph>& subGraph) {
    std::function<std::pair<std::vector<int>, std::shared_ptr<Module>>(EXPRP)> transformFunction;
    if (fortrain) {
        transformFunction =
        [&subGraph](EXPRP source) {
            if (source->get() == nullptr) {
                return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
            }
            std::shared_ptr<Module> m(NN::Utils::ExtractNotRunableOp(source, subGraph));
            if (nullptr != m) {
                m->setName(source->name());
                return std::make_pair(std::vector<int>{}, m);
            }
            auto convExtracted = NN::Utils::ExtractConvolution(source);
            if (convExtracted.weight == nullptr) {
                return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
            }
            std::shared_ptr<Module> module(NN::Conv(convExtracted));
            module->setName(source->name());
            return std::make_pair(std::vector<int>{0}, module);
        };
    } else {
        transformFunction = [&subGraph](EXPRP source) {
            if (source->get() == nullptr) {
                return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
            }
            std::shared_ptr<Module> m(NN::Utils::ExtractNotRunableOp(source, subGraph));
            if (nullptr != m) {
                m->setName(source->name());
                return std::make_pair(std::vector<int>{}, m);
            }
            return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
        };
    }
    return new PipelineModule(inputs, outputs, transformFunction);
}

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
bool PipelineModule::turnQuantize(Module* module, const int bit, NN::FeatureScaleStatMethod featureScaleStatMethod, NN::ScaleUpdateMethod scaleUpdateMethod) {
    if (nullptr == module || module->type() != PIPELINE_MODULE) {
        MNN_ERROR("Invalide module for quantized\n");
        return false;
    }
    ((PipelineModule*)module)->toTrainQuant(bit, featureScaleStatMethod, scaleUpdateMethod);
    return true;
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

void PipelineModule::toTrainQuant(const int bits, NN::FeatureScaleStatMethod featureScaleStatMethod,
                                        NN::ScaleUpdateMethod scaleUpdateMethod) {
    std::vector<int> needEraseIndices;

    for (int i = 0; i < mSubModules.size(); i++) {
        auto& m = mSubModules[i];
        auto& theModule = std::get<0>(m);
        auto moduleType = theModule->type();
        //auto& inputIndices = std::get<1>(m);
        auto& outputIndices = std::get<2>(m);

        if (moduleType == "Conv" && i < mSubModules.size() - 1) {
            auto& p1 = mSubModules[i+1];
            auto p1Module = std::get<0>(p1);
            auto& p1ModuleType = p1Module->type();
            auto& p1InputIndices = std::get<1>(p1);
            auto& p1OutputIndices = std::get<2>(p1);

            auto convOutputCount = countOutputReference(outputIndices);
            bool convSingleOutputReference = ((outputIndices.size() == 1) && (convOutputCount[0] == 1));

            // only conv
            if ((!convSingleOutputReference) || (p1ModuleType == "Conv") ||
                    (p1ModuleType != "BatchNorm" && p1ModuleType != "ReLU" && p1ModuleType != "ReLU6")) {
                theModule.reset(NN::ConvBNReluFused({theModule}, featureScaleStatMethod, scaleUpdateMethod, bits));
                registerModel({theModule});
                continue;
            }
            // conv + bn + ?
            if (p1ModuleType == "BatchNorm") {
                bool convBnConnected = ((convSingleOutputReference) && (p1InputIndices.size() == 1) && (p1InputIndices[0] == outputIndices[0]));
                if (!convBnConnected) {
                    theModule.reset(NN::ConvBNReluFused({theModule}, featureScaleStatMethod, scaleUpdateMethod, bits));
                    registerModel({theModule});
                    continue;
                }

                // last conv + bn
                if (i == mSubModules.size() - 2) {
                    theModule.reset(NN::ConvBNReluFused({theModule, p1Module}, featureScaleStatMethod, scaleUpdateMethod, bits));
                    registerModel({theModule});
                    outputIndices = p1OutputIndices;
                    needEraseIndices.emplace_back(i + 1);
                    continue;
                }
                // maybe there is a relu or relu6 after conv + bn
                auto& p2 = mSubModules[i+2];
                auto& p2Module = std::get<0>(p2);
                auto p2ModuleType = p2Module->type();
                auto& p2InputIndices = std::get<1>(p2);
                auto& p2OutputIndices = std::get<2>(p2);

                auto bnOutputCount = countOutputReference(p1OutputIndices);
                bool bnSingleOutputReference = ((p1OutputIndices.size() == 1) && (bnOutputCount[0] == 1));

                // only conv + bn
                if ((!bnSingleOutputReference) || (p2ModuleType != "ReLU" && p2ModuleType != "ReLU6")) {
                    theModule.reset(NN::ConvBNReluFused({theModule, p1Module}, featureScaleStatMethod, scaleUpdateMethod, bits));
                    registerModel({theModule});
                    outputIndices = p1OutputIndices;
                    needEraseIndices.emplace_back(i + 1);
                    continue;
                } else { // conv + bn + relu or conv + bn + relu6
                    bool convBnReluConnected = ((bnSingleOutputReference) && (p2InputIndices.size() == 1) && (p2InputIndices[0] == p1OutputIndices[0]));
                    if (!convBnReluConnected) {
                        theModule.reset(NN::ConvBNReluFused({theModule, p1Module}, featureScaleStatMethod, scaleUpdateMethod, bits));
                        registerModel({theModule});
                        outputIndices = p1OutputIndices;
                        needEraseIndices.emplace_back(i + 1);
                        continue;
                    }

                    theModule.reset(NN::ConvBNReluFused({theModule, p1Module, p2Module}, featureScaleStatMethod, scaleUpdateMethod, bits));
                    registerModel({theModule});
                    outputIndices = p2OutputIndices;
                    needEraseIndices.emplace_back(i + 1);
                    needEraseIndices.emplace_back(i + 2);
                    continue;
                }
            }
            // conv + relu or conv + relu6
            if (p1ModuleType == "ReLU" || p1ModuleType == "ReLU6") {
                bool convReluConnected = ((convSingleOutputReference) && (p1InputIndices.size() == 1) && (p1InputIndices[0] == outputIndices[0]));
                if (!convReluConnected) {
                    theModule.reset(NN::ConvBNReluFused({theModule}, featureScaleStatMethod, scaleUpdateMethod, bits));
                    registerModel({theModule});
                    continue;
                }

                theModule.reset(NN::ConvBNReluFused({theModule, p1Module}, featureScaleStatMethod, scaleUpdateMethod, bits));
                registerModel({theModule});
                outputIndices = p1OutputIndices;
                needEraseIndices.emplace_back(i + 1);
                continue;
            }
        }

        if (i == mSubModules.size() - 1 && moduleType == "Conv") {
            theModule.reset(NN::ConvBNReluFused({theModule}, featureScaleStatMethod, scaleUpdateMethod, bits));
            registerModel({theModule});
        }
    }

    // erase useless submodules
    const int eraseSize = needEraseIndices.size();
    int alreadyErasedCount = 0;
    for (int i = 0; i < eraseSize; i++) {
        auto position = needEraseIndices[i] - alreadyErasedCount;
        auto type = std::get<0>(mSubModules[position])->type();
        MNN_ASSERT(type == "BatchNorm" || type == "ReLU" || type == "ReLU6");
        mSubModules.erase(mSubModules.begin() + position);
        alreadyErasedCount++;
    }
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

static std::map<std::string, SubGraph> _createSubGraph(const MNN::Net* net, const Module::Config* config) {
    std::map<std::string, SubGraph> subGraphMap;
    auto subGraphs = net->subgraphs();
    if (nullptr == subGraphs) {
        return subGraphMap;
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
            flatbuffers::FlatBufferBuilder builder(1024);
            auto offset = Net::Pack(builder, _tempNet.get());
            builder.Finish(offset);
            // if subgraph contain WhereOp, it's need splite
            auto iter = _tempNet->oplists.begin();
            for (; iter != _tempNet->oplists.end() && iter->get()->type != MNN::OpType_Where; iter++);
            if (config->dynamic || iter != _tempNet->oplists.end()) {
                submodule.reset(PipelineModule::load(subInputs, subOutputs, (const uint8_t*)builder.GetBufferPointer(), builder.GetSize(), config));
            } else {
                submodule.reset(new StaticModule((const uint8_t*)builder.GetBufferPointer(), builder.GetSize(), subInputs, subOutputs, *config));
            }
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
    return subGraphMap;
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

static Module* _createSubModule(const MNN::Net* net, const SubModuleInfo& info, const std::map<std::string, SubGraph>& subs, const Module::Config& config) {
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
    return new StaticModule((const uint8_t*)builder.GetBufferPointer(), builder.GetSize(), inputNames, outputNames, config);
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
    if (!config->dynamic) {
        if (nullptr == subGraphs) {
            auto iter = net->oplists()->begin();
            for (; iter != net->oplists()->end() && iter->type() != OpType_Where; iter++);
            if (iter == net->oplists()->end()) {
                // Has no control flow and WhereOp, can just use static module
                return new StaticModule(buffer, length, inputs, outputs, *config);
            }
        }
    }
    auto subGraphMap = _createSubGraph(net, config);
    if (config->dynamic) {
        // For dynamic mode
        auto varMaps = Variable::loadMap(buffer, length);
        std::vector<VARP> inputVars(inputs.size());
        for (int i=0; i<inputs.size(); ++i) {
            inputVars[i] = varMaps[inputs[i]];
        }
        std::vector<VARP> outputVars(outputs.size());
        for (int i=0; i<outputs.size(); ++i) {
            outputVars[i] = varMaps[outputs[i]];
        }
        return extract(inputVars, outputVars, false, subGraphMap);
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
        subModules[i].reset(_createSubModule(net, subModulesInfo[i], subGraphMap, *config));
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
