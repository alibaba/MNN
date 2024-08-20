//
//  Module.cpp
//  MNN
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "PipelineModule.hpp"
#include "core/FileLoader.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "MNN_generated.h"
#include "Utils.hpp"
#include "RuntimeAttr.hpp"

#include <MNN/AutoTime.hpp>
#ifdef MNN_INTERNAL_ENABLED
#include "internal/auth/ModelAuth.hpp"
#include "internal/logging/Log.hpp"
#include "internal/logging/LogHelper.hpp"
#endif // MNN_INTERNAL_ENABLED

namespace MNN {
namespace Express {
static MNN::Express::Executor::RuntimeManager* _createDefaultRuntimeManager(const Module::Config* config) {
    ScheduleConfig sche_config;
    if(nullptr != config && config->backend != nullptr) {
        sche_config.type = config->backend->type;
        sche_config.backendConfig = config->backend->config;
    } else {
        auto exe = ExecutorScope::Current();
        sche_config.type = exe->getAttr()->firstType;
        sche_config.numThread = 1;
    }
    return Executor::RuntimeManager::createRuntimeManager(sche_config);
}

static Module* loadInternal(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, const std::shared_ptr<MNN::Express::Executor::RuntimeManager> _rtMgr, const Module::Config* config);

class EmptyModule : public Module {
public:
    EmptyModule(const std::vector<Express::VARP>& parameters) {
        for (auto p : parameters) {
            addParameter(p);
        }
    }
    virtual ~EmptyModule() {
        // Do nothing
    }
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override {
        return {};
    }

protected:
    EmptyModule() = default;

    Module* clone(Module::CloneContext* ctx) const override {
        EmptyModule* module(new EmptyModule);
        return this->cloneBaseTo(ctx, module);
    }
};
void Module::destroy(Module* m) {
    if (nullptr != m) {
        delete m;
    }
}

Module* Module::createEmpty(const std::vector<Express::VARP>& parameters) {
    return new EmptyModule(parameters);
}

Express::VARP Module::forward(Express::VARP input) {
    return this->onForward({input})[0];
}
std::vector<Express::VARP> Module::parameters() const {
    std::vector<Express::VARP> result;
    _collectParameters(result);
    return result;
}
bool Module::loadParameters(const std::vector<Express::VARP>& parameters) {
    std::vector<Express::VARP> result;
    _collectParameters(result);
    if (parameters.empty() || parameters.size() != result.size()) {
        MNN_ERROR("Error parameters, empty or parameter size not match \n");
        return false;
    }
    for (int i=0; i<parameters.size(); ++i) {
        if (nullptr != result[i].get()) {
            // Check Origin parameter's size
            auto dstInfo = result[i]->getInfo();
            auto srcInfo = parameters[i]->getInfo();
            if (dstInfo->dim.size() != srcInfo->dim.size() || dstInfo->order != srcInfo->order) {
                MNN_ERROR("Error parameters %d, dim size or order not match \n", i);
                return false;
            }
            if (dstInfo->size != srcInfo->size || dstInfo->type != srcInfo->type) {
                MNN_ERROR("Error parameters %d, size or type not match \n", i);
                return false;
            }
        }
        Variable::replace(result[i], parameters[i]);
    }
    return true;
}
void Module::setIsTraining(const bool isTraining) {
    mIsTraining = isTraining;
    for (auto c : mChildren) {
        c->setIsTraining(isTraining);
    }
}

bool Module::getIsTraining() {
    return mIsTraining;
}

void Module::registerModel(const std::vector<std::shared_ptr<Module>>& children) {
    mChildren.insert(mChildren.begin(), children.begin(), children.end());
}
int Module::addParameter(VARP parameter) {
    auto res = mParameters.size();
    mParameters.emplace_back(parameter);
    return (int)res;
}

void Module::setParameter(Express::VARP parameter, int index) {
    if (index < 0 || index >= mParameters.size()) {
        MNN_ERROR("Module error: index out of range: %d - %d:\n", index, (int)mParameters.size());
        return;
    }
    mParameters[index] = parameter;
}

void Module::_collectParameters(std::vector<Express::VARP>& result) const {
    for (auto p : mParameters) {
        result.push_back(p);
    }
    for (auto c : mChildren) {
        c->_collectParameters(result);
    }
}
void Module::clearCache() {
    for (auto c : mChildren) {
        c->clearCache();
    }
    this->onClearCache();
}

Module* Module::load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const char* fileName, const Module::Config* config) {
    return load(inputs, outputs, fileName, nullptr, config);
}

Module* Module::load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, const Module::Config* config) {
    return load(inputs, outputs, buffer, length, nullptr, config);
}

class NetModule : public Module {
public:
    NetModule(std::shared_ptr<Module> m, std::shared_ptr<Module::Info> info, const MNN::Net* net, size_t size, float costTime) {
        mChildren = {m};
        auto mModule = mChildren[0];
        mInfo = info;
        setType("Net");
#ifdef MNN_INTERNAL_ENABLED
        if (nullptr != net) {
            mLogInfo = logBasicInfo();
            std::string uuid = std::string(net->mnn_uuid() ? net->mnn_uuid()->c_str() : "");
            mLogInfo.emplace("UUID", uuid);
            mLogInfo.emplace("ModelVersion", info->version);
            int backend = MNN_FORWARD_CPU;
            int precision = BackendConfig::Precision_Normal;
            int mode = 1;
            if (info->runTimeManager.get() != nullptr) {
                auto attr = info->runTimeManager->getInside();
                mode = attr->mNumberThread;
                int backendTypes[MNN_FORWARD_ALL];
                info->runTimeManager->getInfo(Interpreter::BACKENDS, &backendTypes);
                backend = backendTypes[0];
                auto config = info->runTimeManager->getBnConfig();
                if (nullptr != config) {
                    precision = config->precision;
                }
            }
            mLogInfo.emplace("Backend",  std::to_string(backend));
            mLogInfo.emplace("Mode",  std::to_string(mode));
            mLogInfo.emplace("Precision", std::to_string(precision));
            if (shouldLog(FREQ_HIGH)) {
                std::map<std::string, std::string> metrics = mLogInfo;
                metrics.emplace("Time", std::to_string(costTime));
                auto sizeInMB = (float)size / 1024.0f / 1024.0f;
                metrics.emplace("ModelSize",  std::to_string(sizeInMB));
                metrics.emplace("API", "Express::Module::NetModule");
                logAsync(metrics);
            }
        }
#endif // MNN_INTERNAL_ENABLED
    }
    virtual ~ NetModule(){
        mChildren.clear();
        mInfo.reset();
        auto exe = ExecutorScope::Current();
        exe->gc(Executor::FULL);
    }

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override {
        auto mModule = mChildren[0];

#ifdef MNN_INTERNAL_ENABLED
        Timer _time;
        auto glo = ExecutorScope::Current();
        glo->getDebugTools()->flops = 0.0f;
#endif
        auto outputs = mModule->onForward(inputs);
#ifdef MNN_INTERNAL_ENABLED
        do {
            if (outputs.empty()) {
                break;
            }
            if (!shouldLog(FREQ_LOW)) {
                break;
            }
            for (auto& v : outputs) {
                auto t = Utils::getTensor(v);
                t->wait(Tensor::MAP_TENSOR_READ, true);
            }
            auto metrics = mLogInfo;
            metrics.emplace("Time", std::to_string((float)_time.durationInUs() / 1000.0f));
            metrics.emplace("API", "NetModule::onForward");
            if (mInfo->runTimeManager.get() != nullptr) {
                float memory = 0.0f;
                mInfo->runTimeManager->getInfo(Interpreter::MEMORY, &memory);
                metrics.emplace("Flops", std::to_string(glo->getDebugTools()->flops));
                metrics.emplace("Memory", std::to_string(memory));
            }
            logAsync(metrics);
            MNN_PRINT("Cost time with log: %f\n", (float)_time.durationInUs() / 1000.0f);
        } while(false);
#endif

        mModule->clearCache();
        return outputs;
    }
    virtual Module* clone(CloneContext* ctx) const override {
        auto mModule = mChildren[0];
        std::shared_ptr<Module> submodule(mModule->clone(ctx));

        NetModule* module(new NetModule(submodule, mInfo, nullptr, 0, 0.0f));
#ifdef MNN_INTERNAL_ENABLED
        module->mLogInfo = mLogInfo;
#endif
        return this->cloneBaseTo(ctx, module);
    }
    const Module::Info* info() const {
        return mInfo.get();
    }

private:
    std::shared_ptr<Module::Info> mInfo;
#ifdef MNN_INTERNAL_ENABLED
    std::map<std::string, std::string> mLogInfo;
#endif
};

const Module::Info* Module::getInfo() const {
    if (mType != "Net") {
        MNN_ERROR("The Module is not load from buffer, can't get info\n");
        return nullptr;
    }
    return ((NetModule*)(this))->info();
}

static void _loadInputs(Module::Info* info, const std::vector<std::string>& inputs, const Net* net) {
    auto type = net->sourceType();
    if (type == NetSource_TENSORFLOW || type == NetSource_TFLITE) {
        info->defaultFormat = NHWC;
    } else {
        info->defaultFormat = NCHW;
    }
    info->inputs.resize(inputs.size());
    std::map<std::string, Variable::Info> allInputs;
    for (int i=0; i<net->oplists()->size(); ++i) {
        auto op = net->oplists()->GetAs<Op>(i);
        if (op->type() == OpType_Input && op->main_as_Input() != nullptr) {
            auto name = net->tensorName()->GetAsString(op->outputIndexes()->data()[0])->str();
            auto inputInfo = op->main_as_Input();
            std::vector<int> dims;
            if (nullptr != inputInfo->dims()) {
                dims.resize(inputInfo->dims()->size());
                for (int v=0; v<dims.size(); ++v) {
                    dims[v] = inputInfo->dims()->data()[v];
                }
            }
            auto dtype = Utils::revertDataType(inputInfo->dtype());
            Variable::Info vinfo;
            vinfo.dim = std::move(dims);
            vinfo.order = Utils::revertFormat(inputInfo->dformat());
            vinfo.type = dtype;
            vinfo.syncSize();
            allInputs.insert(std::make_pair(name, std::move(vinfo)));
        }
    }
    for (int i=0; i<inputs.size(); ++i) {
        auto iter = allInputs.find(inputs[i]);
        if (iter != allInputs.end()) {
            info->inputs[i] = iter->second;
        }
    }
}

Module* Module::load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const char* fileName, const std::shared_ptr<MNN::Express::Executor::RuntimeManager> _rtMgr, const Module::Config* config) {
    AutoStorage<uint8_t> buffer;
    {
        FileLoader loader(fileName, true);
        if (!loader.valid()) {
            MNN_ERROR("Error for open %s\n", fileName);
            return nullptr;
        }
        loader.read();
        if (!loader.valid()) {
            return nullptr;
        }
        loader.merge(buffer);
        if (buffer.get() == nullptr) {
            return nullptr;
        }
    }
    auto rtMgr = _rtMgr;
    if (nullptr == rtMgr.get()) {
        rtMgr.reset(_createDefaultRuntimeManager(config));
    }
    if (rtMgr->getInside()->mExternalFile.empty()) {
        // Set Default externalFile
        rtMgr->setExternalFile(std::string(fileName) + ".weight");
    }
    return loadInternal(inputs, outputs, buffer.get(), buffer.size(), rtMgr, config);
}

Module* Module::load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, const std::shared_ptr<MNN::Express::Executor::RuntimeManager> _rtMgr, const Module::Config* config) {
    auto rtmgr = _rtMgr;
    if (nullptr == rtmgr) {
        rtmgr.reset(_createDefaultRuntimeManager(config));
    }
    return loadInternal(inputs, outputs, buffer, length, rtmgr, config);
}

static Module* loadInternal(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, const std::shared_ptr<MNN::Express::Executor::RuntimeManager> _rtMgr, const Module::Config* config) {
    // Check if runtime is valid
    if (nullptr == _rtMgr || _rtMgr->getInside()->mRuntime.first.empty()) {
        MNN_ERROR("Invalid runtime\n");
        return nullptr;
    }
    bool checkMNNBuffer = true;
    if (nullptr != _rtMgr) {
        checkMNNBuffer = _rtMgr->getInside()->modes.checkNetBuffer;
    }
    if (checkMNNBuffer) {
        flatbuffers::Verifier verify(buffer, length);
        if (false == VerifyNetBuffer(verify)) {
            MNN_PRINT("Invalidate buffer to create MNN Module\n");
            return nullptr;
        }
    }
    // Check Auto Inputs and Outputs
    auto net = GetNet(buffer);
    if (nullptr == net->oplists() || nullptr == net->tensorName()) {
        MNN_ERROR("Invalid net, for null oplist or tensorName\n");
        return nullptr;
    }
    Timer _time;
    std::shared_ptr<Module::Info> info(new Module::Info);
    if (net->extraInfo() && net->extraInfo()->version()) {
        info->version = net->extraInfo()->version()->str();
    }
    auto rtMgr = _rtMgr;
    Module::Config defaultConfig;
    if (nullptr == config) {
        config = &defaultConfig;
    }
    info->inputNames = inputs;
    info->outputNames = outputs;
    if ((!inputs.empty()) && (!outputs.empty())) {
        _loadInputs(info.get(), inputs, net);
        info->runTimeManager = rtMgr;
        std::shared_ptr<Module> m(PipelineModule::load(inputs, outputs, buffer, length, rtMgr, config));
        return new NetModule(m, info, net, length, (float)_time.durationInUs() / 1000.0f);
    }
    std::set<int> inputIdx, outputIdx, realInput, realOutput;
    for (int i=0; i< net->oplists()->size(); ++i) {
        auto op = net->oplists()->GetAs<Op>(i);
        if (nullptr != op->inputIndexes()) {
            auto data = op->inputIndexes()->data();
            auto size = op->inputIndexes()->size();
            for (int j=0; j<size; ++j) {
                inputIdx.insert(data[j]);
            }
        }
        if (nullptr != op->outputIndexes()) {
            auto data = op->outputIndexes()->data();
            auto size = op->outputIndexes()->size();
            for (int j=0; j<size; ++j) {
                outputIdx.insert(data[j]);
                if (op->type() == OpType_Input) {
                    realInput.insert(data[j]);
                }
            }
        }
    }
    if (info->inputNames.empty()) {
        for (auto index : realInput) {
            info->inputNames.emplace_back(net->tensorName()->GetAsString(index)->str());
        }
    }
    if (info->outputNames.empty()) {
        if (nullptr != net->outputName()) {
            for (int i=0; i<net->outputName()->size(); ++i) {
                info->outputNames.emplace_back(net->outputName()->GetAsString(i)->str());
            }
        } else {
            std::set_difference(outputIdx.begin(), outputIdx.end(), inputIdx.begin(), inputIdx.end(), std::inserter(realOutput, realOutput.begin()));
            for (auto index : realOutput) {
                info->outputNames.emplace_back(net->tensorName()->GetAsString(index)->str());
            }
        }
    }
    std::shared_ptr<Module> m(PipelineModule::load(info->inputNames, info->outputNames, buffer, length, rtMgr, config));
    _loadInputs(info.get(), info->inputNames, net);
    info->runTimeManager = rtMgr;
    return new NetModule(m, info, net, length, (float)_time.durationInUs() / 1000.0f);
}

EXPRP Module::CloneContext::getOrClone(EXPRP expr) {
    auto it = mExprMap.find(expr.get());
    if (it == mExprMap.end()) {
        EXPRP replica;
        if (expr->get() == nullptr) {
            VARP var = Variable::create(expr);
            Variable::Info info(*var->getInfo());
            replica = Expr::create(std::move(info), var->readMap<void>(), expr->inputType(),
                                   (expr->inputType() != VARP::CONSTANT) ? Expr::COPY : Expr::REF);
        } else {
            std::vector<VARP> inputs;
            for (auto& input: expr->inputs()) {
                inputs.emplace_back(getOrClone(input));
            }
            replica = Expr::create(expr->extra(), std::move(inputs), expr->outputSize());
        }
        replica->setName(expr->name());
        it = mExprMap.emplace(expr.get(), replica).first;
    }
    return it->second;
}

VARP Module::CloneContext::getOrClone(VARP var) {
    auto it = mVarMap.find(var.get());
    if (it == mVarMap.end()) {
        auto expr = var->expr();
        VARP replica = Variable::create(getOrClone(expr.first), expr.second);
        it = mVarMap.emplace(var.get(), replica).first;
    }
    return it->second;
}

Module* Module::clone(const Module* module, const bool shareParams) {
    CloneContext context(shareParams);
    return module->clone(&context);
}

Module* Module::cloneBaseTo(CloneContext* ctx, Module* module) const {
    for (const Express::VARP& var : mParameters) {
        module->mParameters.push_back(ctx->getOrClone(var));
    }
    module->mIsTraining = mIsTraining;
    module->mName = mName;
    module->mType = mType;
    return module;
}

Module* Module::extract(std::vector<Express::VARP> inputs, std::vector<Express::VARP> outputs, bool fortrain, const std::map<std::string, SubGraph>& subGraph) {
    return new PipelineModule(inputs, outputs);
}
int Module::traceOrOptimize(Interpreter::SessionMode stage) {
    for (auto& m : mChildren) {
        m->traceOrOptimize(stage);
    }
    return this->onOptimize(stage);
}


} // namespace Express
} // namespace MNN
