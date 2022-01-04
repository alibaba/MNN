//
//  Module.cpp
//  MNN
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "PipelineModule.hpp"
#include "core/FileLoader.hpp"
#include "MNN_generated.h"
#include "Utils.hpp"

#ifdef MNN_INTERNAL_ENABLED
#include "internal/auth/ModelAuth.hpp"
#include "internal/logging/Log.hpp"
#include "internal/logging/LogHelper.hpp"
#endif // MNN_INTERNAL_ENABLED

namespace MNN {
namespace Express {

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


Module* Module::load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const char* fileName, const std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Module::Config* config) {
    AutoStorage<uint8_t> buffer;
    {
        FileLoader loader(fileName);
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
    return load(inputs, outputs, buffer.get(), buffer.size(), rtMgr, config);
}
class NetModule : public Module {
public:
    NetModule(std::shared_ptr<Module> m, std::shared_ptr<Module::Info> info) {
        mModule = m;
        mInfo = info;
        setType("Net");
    }
    virtual ~ NetModule(){}

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override {
        return mModule->onForward(inputs);
    }
    virtual Module* clone(CloneContext* ctx) const override {
        NetModule* module(new NetModule(mModule, mInfo));
        return this->cloneBaseTo(ctx, module);
    }
    const Module::Info* info() const {
        return mInfo.get();
    }
private:
    std::shared_ptr<Module> mModule;
    std::shared_ptr<Module::Info> mInfo;
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

Module* Module::load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, const std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Module::Config* config) {
    // Check if runtime is valid
    if (nullptr != rtMgr && rtMgr->getRuntimeInfo().first.empty()) {
        MNN_ERROR("Invalid runtime\n");
        return nullptr;
    }
    // Check Auto Inputs and Outputs
    auto net = GetNet(buffer);
    if (nullptr == net->oplists() || nullptr == net->tensorName()) {
        MNN_ERROR("Invalid net, for null oplist or tensorName\n");
        return nullptr;
    }

#ifdef MNN_INTERNAL_ENABLED
    std::string bizCode = std::string(net->bizCode() ? net->bizCode()->c_str() : "");
    std::string uuid = std::string(net->mnn_uuid() ? net->mnn_uuid()->c_str() : "");

    if (!authenticateModel(net)) {
        MNN_ERROR("Model authentication failed.\n");

        std::map<std::string, std::string> metrics;
        metrics.emplace("Model_UUID", uuid);
        metrics.emplace("Model_BizCode", bizCode);
        metrics.emplace("Event", "AUTH_FAILURE");
        metrics.emplace("Backend", config && config->backend ? std::to_string(config->backend->type) : std::to_string(MNN_FORWARD_CPU));
        metrics.emplace("Precision", config && config->backend && config->backend->config ? std::to_string(config->backend->config->precision) : std::to_string(BackendConfig::Precision_Normal));
        metrics.emplace("API", "Express::Module::load");
        auto basicMetrics = getBasicLoggingData();
        metrics.insert(basicMetrics.begin(), basicMetrics.end());
        logAsync(metrics);

        return nullptr;
    }
    std::map<std::string, std::string> metrics;
    metrics.emplace("Model_UUID", uuid);
    metrics.emplace("Model_BizCode", bizCode);
    metrics.emplace("Event", "AUTH_SUCCESS");
    metrics.emplace("Backend", config && config->backend ? std::to_string(config->backend->type) : std::to_string(MNN_FORWARD_CPU));
    metrics.emplace("Precision", config && config->backend && config->backend->config ? std::to_string(config->backend->config->precision) : std::to_string(BackendConfig::Precision_Normal));
    metrics.emplace("API", "Express::Module::load");
    auto basicMetrics = getBasicLoggingData();
    metrics.insert(basicMetrics.begin(), basicMetrics.end());
    logAsync(metrics);
#endif // MNN_INTERNAL_ENABLED

    std::shared_ptr<Info> info(new Info);
    if ((!inputs.empty()) && (!outputs.empty())) {
        _loadInputs(info.get(), inputs, net);
        info->runTimeManager = rtMgr;
        std::shared_ptr<Module> m(PipelineModule::load(inputs, outputs, buffer, length, rtMgr, config));
        return new NetModule(m, info);
    }
    std::vector<std::string> newInputs = inputs;
    std::vector<std::string> newOutputs = outputs;
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
    std::set_difference(outputIdx.begin(), outputIdx.end(), inputIdx.begin(), inputIdx.end(), std::inserter(realOutput, realOutput.begin()));
    if (newInputs.empty()) {
        for (auto index : realInput) {
            newInputs.emplace_back(net->tensorName()->GetAsString(index)->str());
        }
    }
    if (newOutputs.empty()) {
        for (auto index : realOutput) {
            newOutputs.emplace_back(net->tensorName()->GetAsString(index)->str());
        }
    }
    std::shared_ptr<Module> m(PipelineModule::load(newInputs, newOutputs, buffer, length, rtMgr, config));
    _loadInputs(info.get(), newInputs, net);
    info->runTimeManager = rtMgr;
    return new NetModule(m, info);
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

} // namespace Express
} // namespace MNN
