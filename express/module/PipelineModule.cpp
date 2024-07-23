//
//  PipelineModule.cpp
//  MNN
//
//  Created by MNN on 2020/01/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "PipelineModule.hpp"
#include <set>
#include <vector>
#include "StaticModule.hpp"
#include "IfModule.hpp"
#include "WhileModule.hpp"
#include "NMSModule.hpp"
#include "Utils.hpp"
#include "core/Backend.hpp"
#include "core/WrapExecution.hpp"
#include "core/FileLoader.hpp"
#include "utils/InitNet.hpp"
#include "RuntimeAttr.hpp"
#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"

#include <MNN/expr/ExecutorScope.hpp>
using namespace MNN::Express;
namespace MNN {
namespace Express {
//#define DYNAMIC
//#define MNN_PIPELINE_MODULE_DEBUG

ExprModule::ExprModule(EXPRP expr) {
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
    // TODO: Optimize the logic
    if (!mExpr->mCanDecompose) {
        ExecutorScope::Current()->setLazyComputeMode(Executor::LAZY_CONTENT);
    }
}

std::vector<VARP> ExprModule::onForward(const std::vector<VARP>& inputs) {
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
    if (!mExpr->mCanDecompose) {
        // Set tensor shape from net
        newExpr->mCanDecompose = false;
        for (int index = 0; index < mExpr->outputSize(); ++index) {
            TensorUtils::copyShape(mExpr->inside()->mOutputTensors[index], newExpr->inside()->mOutputTensors[index], true, true);
            Utils::copyTensorToInfo(newExpr->inside()->mOutputInfos.data() + index, newExpr->inside()->mOutputTensors[index]);
        }
    }
    for (int i = 0; i < mExpr->outputSize(); ++i) {
        outputVars.emplace_back(Variable::create(newExpr, i));
    }
    return outputVars;
}

Module* ExprModule::clone(CloneContext* ctx) const {
    ExprModule* module(new ExprModule(ctx->getOrClone(mExpr)));
    for (const VARP& var : mInputs) {
        module->mInputs.push_back(ctx->getOrClone(var));
    }
    module->mInputIndexes = mInputIndexes;
    return this->cloneBaseTo(ctx, module);
}

PipelineModule::PipelineModule(std::vector<VARP> inputs, std::vector<VARP> outputs, const Transformer& transformFunction) {
    setType(PIPELINE_MODULE);
    std::map<EXPRP, int> inputExpr;
    for (int i=0; i<inputs.size(); ++i) {
        auto expr = inputs[i]->expr().first;
        inputExpr.insert(std::make_pair(expr, i));
    }
    std::vector<EXPRP> executeOrder = Variable::getExecuteOrder(outputs);
    // Set Indexes
    std::map<EXPRP, int> indexes;
    mInputSize = inputs.size();
    int currentIndexes = inputs.size();
    for (auto expr : executeOrder) {
        if (inputExpr.find(expr) != inputExpr.end()) {
            indexes[expr] = inputExpr[expr];
            continue;
        }
        indexes[expr] = currentIndexes;
        currentIndexes += expr->outputSize();
    }
    std::set<EXPRP> inputSets;
    mStackSize = currentIndexes;
    for (auto v : inputs) {
        auto inputExpr = v->expr();
        inputSets.insert(inputExpr.first);
    }
    mOutputIndex.clear();
    for (auto output : outputs) {
        auto outputExpr = output->expr();
        mOutputIndex.emplace_back(indexes[outputExpr.first] + outputExpr.second);
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
    for (int i = 0; i < mInitVars.size(); ++i) {
        mStack[i + mInputSize] = mInitVars[i];
    }
    MNN_ASSERT(mInputSize == inputs.size());
    for (int i = 0; i < mInputSize; ++i) {
        mStack[i] = inputs[i];
    }
    for (int index = 0; index < mSubModules.size(); ++index) {
        auto& m = mSubModules[index];
        std::vector<VARP> tempInputs(std::get<1>(m).size());
        for (int i = 0; i < tempInputs.size(); ++i) {
            auto stackInput = std::get<1>(m)[i];
            tempInputs[i] = mStack[stackInput];
            MNN_ASSERT(nullptr != tempInputs[i]);
        }
        std::vector<VARP> tempOutputs = std::get<0>(m)->onForward(tempInputs);
        if(tempOutputs.size() != std::get<2>(m).size()) {
            // Execute has error
            return {};
        }
        for (int i = 0; i < tempOutputs.size(); ++i) {
            mStack[std::get<2>(m)[i]] = tempOutputs[i];
            MNN_ASSERT(nullptr != tempOutputs[i]);
        }
    }
    std::vector<VARP> outputs(mOutputIndex.size());
    for (int i = 0; i < mOutputIndex.size(); ++i) {
        outputs[i] = mStack[mOutputIndex[i]];
    }
    return outputs;
}
void PipelineModule::onClearCache() {
    // Do nothing
}

void PipelineModule::_createSubGraph(const MNN::Net* net, std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Module::Config* config, std::map<std::string, SubGraph>& subGraphMap) {
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
#ifdef MNN_PIPELINE_MODULE_DEBUG
        for (auto& s : subOutputs) {
            FUNC_PRINT_ALL(s.c_str(), s);
        }
        FUNC_PRINT_ALL(graph->name()->c_str(), s);
#endif
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
            std::shared_ptr<BufferStorage> bufferStorage(new BufferStorage);
            bufferStorage->storage = builder.ReleaseRaw(bufferStorage->allocated_size, bufferStorage->offset);
            submodule.reset(PipelineModule::load(subInputs, subOutputs, bufferStorage, rtMgr, config, subGraphMap));
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
    bool isBreak = false;
};
static void _computeTensorMask(SubModuleInfo& m, const Net* net) {
    /**Compute All SubModule's inputs and outputs*/
    // 0: not use, 1: input, 2: output, 3: mid, 4: valid output
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
}

static bool isBreakOp(const Op* op) {
    bool isWhileControlflow = false;
    if (op->type() == OpType_While && op->main_as_WhileParam() != nullptr) {
        isWhileControlflow = true;
    }
    if (op->type() == OpType_If || isWhileControlflow || op->type() == OpType_Where || op->type() == OpType_Segment || op->type() == OpType_Unique || op->type() == OpType_NonMaxSuppressionV2) {
        return true;
    }
    return false;
}

static std::vector<int> _collectNeededOps(const MNN::Net* net, const std::set<int>& inputIndexes, const std::set<int>& outputIndexes) {
    // 0: not set, 1: output, 2:input
    std::vector<int> tensorMask(net->tensorName()->size());
    ::memset(tensorMask.data(), 0, tensorMask.size() * sizeof(int));

    // 0: use, 1: no use
    std::vector<int> opMask(net->oplists()->size());
    ::memset(opMask.data(), 0, opMask.size() * sizeof(int));

    // Set Initial Status
    for (auto v : outputIndexes) {
        tensorMask[v] = 1;
    }
    for (auto v : inputIndexes) {
        // If both input/output, set as input
        tensorMask[v] = 2;
    }
    bool change = false;
    do {
        change = false;
        for (int i=0; i<opMask.size(); ++i) {
            if (opMask[i] > 0) {
                continue;
            }
            auto op = net->oplists()->GetAs<Op>(i);
            if (nullptr != op->outputIndexes()) {
                for (int j=0; j<op->outputIndexes()->size(); ++j) {
                    auto index = op->outputIndexes()->data()[j];
                    if (tensorMask[index] == 1) {
                        opMask[i] = 1;
                        change = true;
                    }
                }
            }
            if (nullptr != op->inputIndexes() && opMask[i]) {
                for (int j=0; j<op->inputIndexes()->size(); ++j) {
                    auto index = op->inputIndexes()->data()[j];
                    if (tensorMask[index] != 2) {
                        tensorMask[index] = 1;
                    }
                }
            }
        }
    } while (change);

    std::vector<int> ops;
    for (int i=0; i<opMask.size(); ++i) {
        if (opMask[i] > 0) {
            auto op = net->oplists()->GetAs<Op>(i);
            if (needComputeOp(op)) {
                ops.emplace_back(i);
                continue;
            }
        }
    }
    return ops;
}

static std::vector<SubModuleInfo> _createSubModuleInfo(std::shared_ptr<BufferStorage> bufferStorage, const std::set<int>& inputIndexes, const std::set<int>& outputIndexes, const std::set<int>& noComputeIndexes, std::shared_ptr<Schedule::ScheduleInfo> sharedConst) {
    std::vector<SubModuleInfo> submodule;
    auto net = flatbuffers::GetRoot<Net>(bufferStorage->buffer());
    auto selectOps = _collectNeededOps(net, inputIndexes, outputIndexes);

    // Separate the graph to serveral submodule
    SubModuleInfo current;
    for (int si=0; si<selectOps.size(); ++si) {
        auto i = selectOps[si];
        auto op = net->oplists()->GetAs<Op>(i);
        if (isBreakOp(op)) {
            // TODO: Don't need split segment
            if (current.opList.size() > 0) {
                // Not empty
                // Init tensormask
                _computeTensorMask(current, net);
                submodule.emplace_back(std::move(current));
            }
            SubModuleInfo controlOp;
            controlOp.opList = {i};
            controlOp.isBreak = true;
            if (nullptr != op->inputIndexes()) {
                controlOp.inputs.resize(op->inputIndexes()->size());
                ::memcpy(controlOp.inputs.data(), op->inputIndexes()->data(), controlOp.inputs.size() * sizeof(int));
            }
            if (nullptr != op->outputIndexes()) {
                controlOp.outputs.resize(op->outputIndexes()->size());
                ::memcpy(controlOp.outputs.data(), op->outputIndexes()->data(), controlOp.outputs.size() * sizeof(int));
            }
            submodule.emplace_back(std::move(controlOp));
            continue;
        }
        current.opList.emplace_back(i);
    }
    if (!current.opList.empty()) {
        _computeTensorMask(current, net);
        submodule.emplace_back(std::move(current));
    }
    for (int moduleIndex=0; moduleIndex < submodule.size(); ++moduleIndex) {
        auto& m = submodule[moduleIndex];
        // Compute input / output
        if (!m.isBreak) {
            for (int i=0; i<m.tensorMask.size(); ++i) {
                if (0 == m.tensorMask[i]) {
                    continue;
                }
                if (1 == m.tensorMask[i]) {
                    if (noComputeIndexes.find(i) != noComputeIndexes.end()) {
                        continue;
                    }
                    m.inputs.emplace_back(i);
                    continue;
                }
                if (2 == m.tensorMask[i]) {
                    m.outputs.emplace_back(i);
                    continue;
                }
                if (3 == m.tensorMask[i]) {
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
            if (noComputeIndexes.find(index) != noComputeIndexes.end()) {
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

struct ModuleRuntimeConfig {
    bool needGeometry;
    RuntimeInfo rt;
    Backend::Info compute;
    const BackendConfig* userConfig = nullptr;
    Session::ModeGroup modes;
    std::string externalFile;
};

static Module* _createSubModule(std::shared_ptr<BufferStorage> bufferStorage, const SubModuleInfo& info, const std::map<std::string, SubGraph>& subs, std::shared_ptr<Schedule::ScheduleInfo> sharedConst, const Module::Config& config, const ModuleRuntimeConfig& runtimeConfig) {
    auto net = flatbuffers::GetRoot<Net>(bufferStorage->buffer());
    if (1 == info.opList.size()) {
        auto op = net->oplists()->GetAs<Op>(info.opList[0]);
        if (OpType_If == op->type()) {
            return IfModule::create(op, subs);
        }
        if (OpType_While == op->type() && op->main_type() != OpParameter_LoopParam) {
            return WhileModule::create(op, subs);
        }
        if (OpType_NonMaxSuppressionV2 == op->type()) {
            return NMSModule::create(op);
        }
        // MNN_ASSERT(false);
    }
    Schedule::ScheduleInfo scheduleInfo;
    scheduleInfo.externalWeightPath = runtimeConfig.externalFile;
    scheduleInfo.defaultBackend = sharedConst->defaultBackend;
    scheduleInfo.constReplaceBackend = sharedConst->constReplaceBackend;
    scheduleInfo.allTensors = sharedConst->allTensors;
    scheduleInfo.validForResize = initTensors(scheduleInfo.allTensors, net);
    std::vector<Schedule::OpCacheInfo> oplists;
    std::vector<const Op*> ops;
    ops.reserve(info.opList.size());
    for (auto opIndex : info.opList) {
        ops.emplace_back(net->oplists()->GetAs<Op>(opIndex));
    }
    initPipelineInfosFromOps(oplists, ops, scheduleInfo.allTensors);
    int breakIndex = GeometryComputerUtils::buildConstantTensors(oplists);
    if (breakIndex >= 0) {
        scheduleInfo.needInputContentForShape = true;
    }
    auto rt = runtimeConfig.rt;
    auto modes = runtimeConfig.modes;
    Schedule::BackendCache bnCache;
    Backend::Info compute = runtimeConfig.compute;
    if (nullptr != runtimeConfig.userConfig) {
        bnCache.config = *runtimeConfig.userConfig;
        compute.user      = &bnCache.config;
    } else {
        compute.user      = nullptr;
    }
    bnCache.info = std::move(compute);
    bnCache.needComputeGeometry = runtimeConfig.needGeometry;

    scheduleInfo.pipelineInfo.emplace_back(std::make_pair(std::move(bnCache), std::move(oplists)));

    std::vector<std::shared_ptr<BufferStorage>> buffers = {bufferStorage};

    return new StaticModule(info.inputs, info.outputs, std::move(buffers), std::move(scheduleInfo), sharedConst, std::move(modes), std::move(rt), config);
}

Module* PipelineModule::load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, const std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Module::Config* config) {
    // Create Subgraph
    auto net = GetNet(buffer);
    if (nullptr == net->oplists() || nullptr == net->tensorName()) {
        MNN_ERROR("Invalid net, for null oplist or tensorName\n");
        return nullptr;
    }
    Module::Config defaultConfig;
    if (nullptr == config) {
        config = &defaultConfig;
    }
    auto subGraphs = net->subgraphs();
    if (config->dynamic) {
        // TODO: Support subgraph
        if (nullptr == subGraphs) {
            auto varMap = MNN::Express::Variable::loadMap(buffer, length);
            std::vector<MNN::Express::VARP> inputsVar(inputs.size());
            for (int i=0; i<inputs.size(); ++i) {
                inputsVar[i] = varMap[inputs[i]];
            }
            std::vector<MNN::Express::VARP> outputsVar(outputs.size());
            for (int i=0; i<outputs.size(); ++i) {
                outputsVar[i] = varMap[outputs[i]];
            }
            return extract(inputsVar, outputsVar, false);
        } else {
            MNN_ERROR("Don't support subgraph for dynamic load, turn back to static load\n");
        }
    }
    std::map<std::string, SubGraph> subGraphMap;
    _createSubGraph(net, rtMgr, config, subGraphMap);
    std::shared_ptr<BufferStorage> bufferStorage(new BufferStorage);
    bufferStorage->storage = new uint8_t[length];
    ::memcpy(bufferStorage->storage, buffer, length);
    bufferStorage->offset = 0;
    bufferStorage->allocated_size = length;
    return load(inputs, outputs, bufferStorage, rtMgr, config, subGraphMap);
}

Module* PipelineModule::load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, std::shared_ptr<BufferStorage> bufferStorage, std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Module::Config* config, std::map<std::string, SubGraph>& subGraphMap) {
    MNN_ASSERT(nullptr != rtMgr);
    std::shared_ptr<Schedule::ScheduleInfo> sharedConst;
    auto buffer = bufferStorage->buffer();
    auto length = bufferStorage->size();
    auto net = GetNet(buffer);
    bool needGeometry = net->usage() != Usage_INFERENCE_STATIC;
    // Extra Const Tensors
    sharedConst.reset(new Schedule::ScheduleInfo);
    auto curExe = ExecutorScope::Current();
    bool permitCodeGen = false;
    std::shared_ptr<ModuleRuntimeConfig> modRuntimeCfgPtr(new ModuleRuntimeConfig);
    if (!rtMgr->getInside()->mExternalFile.empty()) {
        modRuntimeCfgPtr->externalFile = rtMgr->getInside()->mExternalFile;
    }
    permitCodeGen = rtMgr->getInside()->modes.codegenMode == Interpreter::Session_Codegen_Enable;
    std::shared_ptr<Backend> defaultBackend = curExe->getAttr()->constantBackend;
    std::vector<std::shared_ptr<Tensor>> allTensors;
    sharedConst->allTensors.resize(net->tensorName()->size());
    sharedConst->defaultBackend = defaultBackend;
    ModuleRuntimeConfig& modRuntime = *modRuntimeCfgPtr;
    modRuntime.needGeometry = needGeometry;
    {
        modRuntime.modes = rtMgr->getInside()->modes;
        modRuntime.rt = rtMgr->getInside()->mRuntime;
        modRuntime.externalFile = rtMgr->getInside()->mExternalFile;
        modRuntime.userConfig = &rtMgr->getInside()->mConfig;
        modRuntime.compute.type      = modRuntime.rt.first.begin()->first;
        modRuntime.compute.numThread = 1;
        // set allocator type
        modRuntime.rt.first.begin()->second->setRuntimeHint(rtMgr->getInside()->modes.runtimeHint);
        // set winograd memory type
        modRuntime.rt.second->setRuntimeHint(rtMgr->getInside()->modes.runtimeHint);
    }
    auto& rt = modRuntime.rt;
    auto firstRt = rt.first[modRuntime.compute.type];
    sharedConst->constReplaceBackend.reset(firstRt->onCreate(modRuntime.userConfig));
    ErrorCode code = NO_ERROR;
    std::set<int> noneedComputeIndexes;
    {
        FileLoader fileLoader(modRuntimeCfgPtr->externalFile.c_str());
        initConstTensors(sharedConst->allTensors, net, defaultBackend.get(), code, &fileLoader);
    }
    if (NO_ERROR != code) {
        MNN_ERROR("Alloc memory for const tensor error\n");
        return nullptr;
    }
    for (int i=0; i<sharedConst->allTensors.size(); ++i) {
        if (sharedConst->allTensors[i].get() != nullptr) {
            noneedComputeIndexes.insert(i);
        }
    }

    std::map<int, VARP> initVars;
    std::set<int> inputIndexes;
    std::set<int> outputIndexes;
    std::map<int, int> stackMap;
    std::map<std::string, int> outputIndexesMap;
    for (int i=0; i<net->tensorName()->size(); ++i) {
        auto tname = net->tensorName()->GetAsString(i)->str();
        for (int j=0; j<inputs.size(); ++j) {
            if (tname == inputs[j]) {
                inputIndexes.emplace(i);
                stackMap.insert(std::make_pair(i, j));
                break;
            }
        }
        for (int j=0; j<outputs.size(); ++j) {
            if (tname == outputs[j]) {
                outputIndexes.emplace(i);
                outputIndexesMap.insert(std::make_pair(tname, i));
                break;
            }
        }
    }
    if (outputIndexesMap.size() != outputs.size()) {
        MNN_ERROR("PipelineModule:: Can't find enough output from the model, finded is:\n");
        for (auto& iter : outputIndexesMap) {
            MNN_ERROR("[ %s ] ", iter.first.c_str());
        }
        MNN_ERROR("\n");
        return nullptr;
    }
    for (auto index : noneedComputeIndexes) {
        auto tensor = Tensor::clone(sharedConst->allTensors[index].get());
        auto constVar = Variable::create(Expr::create(tensor, true));
        initVars.insert(std::make_pair(index, constVar));
    }
    auto subModulesInfo = _createSubModuleInfo(bufferStorage, inputIndexes, outputIndexes, noneedComputeIndexes, sharedConst);
    std::vector<std::shared_ptr<Module>> subModules(subModulesInfo.size());
    for (int i=0; i<subModulesInfo.size(); ++i) {
        subModules[i].reset(_createSubModule(bufferStorage, subModulesInfo[i], subGraphMap, sharedConst, *config, modRuntime));
    }
    auto result = new PipelineModule;
    result->mInputSize = inputs.size();
    /**
     Compute:
     std::vector<std::tuple<std::shared_ptr<Module>, std::vector<int>, std::vector<int>>> mSubModules;
     std::vector<int> mInputIndexes;
     std::vector<int> mOutputIndexes;
     int mStackSize = 0;
     */
    // Make Stack, first: origin, second: new
    int stackIndex = result->mInputSize;
    for (auto& p : initVars) {
        stackMap.insert(std::make_pair(p.first, stackIndex));
        result->mInitVars.emplace_back(p.second);
        stackIndex++;
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
    result->mOutputIndex.resize(outputs.size());
    for (int i=0; i<outputs.size(); ++i) {
        auto index = outputIndexesMap[outputs[i]];
        MNN_ASSERT(stackMap.find(index) != stackMap.end());
        auto stackI = stackMap[index];
        result->mOutputIndex[i] = stackI;
    }
    result->mStackSize = stackIndex;
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
    result->registerModel(subModules);
    result->mSharedConst = sharedConst;
    if (!permitCodeGen) {
        // Prereplace const tensor
        auto curBackend = sharedConst->constReplaceBackend.get();
        if (sharedConst->constReplaceBackend->type() != sharedConst->defaultBackend->type()) {
            for (auto& t : sharedConst->allTensors) {
                if (nullptr == t.get()) {
                    continue;
                }
                auto des = TensorUtils::getDescribe(t.get());
                if (des->isMutable) {
                    continue;
                }
                if (!WrapExecution::needWrap(t.get(), curBackend)) {
                    continue;
                }
                if (des->stageMask & Tensor::InsideDescribe::GEOMETRY_STAGE) {
                    continue;
                }
                if (des->stageMask & Tensor::InsideDescribe::CONVERTED_STAGE) {
                    continue;
                }
                std::shared_ptr<Tensor> wrapTensor = WrapExecution::makeCopyTensor(t.get(), curBackend);
                auto outDes = TensorUtils::getDescribe(wrapTensor.get());
                outDes->usage = des->usage;
                auto tempRes = WrapExecution::allocAndCopy(curBackend, t.get(), wrapTensor.get());
                if (!tempRes) {
                    continue;
                }
                outDes->stageMask |= Tensor::InsideDescribe::CONVERTED_STAGE;
                WrapExecution::copyReplaceTensor(wrapTensor.get(), t.get());
            }
        }
    }
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
    module->mInputSize = mInputSize;
    module->mOutputIndex = mOutputIndex;
    module->mStackSize = mStackSize;
    module->mInitVars = mInitVars;
    module->mSharedConst = mSharedConst;
    return this->cloneBaseTo(ctx, module);
}


} // namespace Express
} // namespace MNN
