//
//  Interpreter.cpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <stdio.h>
#include <MNN/Interpreter.hpp>
#include <algorithm>
#include <mutex>
#include <vector>
#include "MNN_generated.h"
#include "core/AutoStorage.h"
#include "core/FileLoader.hpp"
#include "core/Pipeline.hpp"
#include "core/RuntimeFactory.hpp"
#include "core/Session.hpp"
#include <MNN/AutoTime.hpp>

#ifdef MNN_INTERNAL_ENABLED
#include "internal/logging/Log.hpp"
#include "internal/logging/LogHelper.hpp"
#endif // MNN_INTERNAL_ENABLED

namespace MNN {

struct Content {
    AutoStorage<uint8_t> buffer;
    const Net* net = nullptr;
    std::vector<std::unique_ptr<Session>> sessions;
    std::map<Tensor*, const Session*> tensorMap;
    Session::ModeGroup modes;
    AutoStorage<uint8_t> cacheBuffer;
    std::string cacheFile;
    std::mutex lock;
    size_t lastCacheSize = 0;
    std::string bizCode;
    std::string uuid;
    std::string externalFile;
#ifdef MNN_INTERNAL_ENABLED
    std::map<std::string, std::string> basicLogginData;
    std::map<const Session*, std::tuple<int, int>> sessionInfo;
#endif
};

const char* getVersion() {
    return MNN_VERSION;
}

static void writeCacheFile(const Content *net, std::pair<const void*, size_t> buffer) {
    bool res = FileLoader::write(net->cacheFile.c_str(), buffer);
    if (!res) {
        MNN_ERROR("Write Cache File error!\n");
        return;
    }
}

static Content* loadModelFile(const char* file) {
    if (nullptr == file) {
        MNN_PRINT("NULL file for create interpreter\n");
        return nullptr;
    }
    std::unique_ptr<FileLoader> loader(new FileLoader(file, true));
    if (!loader->valid()) {
        MNN_PRINT("Create interpreter failed, open %s error\n", file);
        return nullptr;
    }
    bool result = loader->read();
    if (!result) {
        MNN_PRINT("Read file error\n");
        return nullptr;
    }
    if (loader->size() == 0) {
        MNN_PRINT("Create interpreter failed, %s is empty\n", file);
        return nullptr;
    }
    auto net     = new Content;
    bool success = loader->merge(net->buffer);
    if (!success) {
        return nullptr;
    }
    loader.reset();
    return net;
}

Interpreter* Interpreter::createFromFile(const char* file) {
    Content* net = loadModelFile(file);
    if (nullptr == net) {
        return nullptr;
    }
    // Set Default externalFile
    net->externalFile = std::string(file) + ".weight";

    return createFromBufferInternal(net, true);
}
Interpreter* Interpreter::createFromBuffer(const void* buffer, size_t size) {
    if (nullptr == buffer || 0 == size) {
        MNN_PRINT("Buffer is null for create interpreter\n");
        return nullptr;
    }
    auto net = new Content;
    net->buffer.reset((int)size);
    if (nullptr == net->buffer.get()) {
        MNN_ERROR("Memory not enought!\n");
        return nullptr;
    }
    ::memcpy(net->buffer.get(), buffer, size);

    return createFromBufferInternal(net, true);
}

Interpreter* Interpreter::createFromBufferInternal(Content* net, bool enforceAuth) {
    if (nullptr == net) {
        MNN_PRINT("Buffer is null for create interpreter\n");
        return nullptr;
    }
#ifndef MNN_BUILD_MINI
    flatbuffers::Verifier verify((const uint8_t*)(net->buffer.get()), net->buffer.size());
    if (false == VerifyNetBuffer(verify)) {
        MNN_PRINT("Invalidate buffer to create interpreter\n");
        delete net;
        return nullptr;
    }
#endif
    net->net = GetNet(net->buffer.get());
    if (nullptr == net->net->oplists()) {
        MNN_ERROR("Model has no oplist\n");
        delete net;
        return nullptr;
    }
    int opSize = net->net->oplists()->size();
    for (int i = 0; i < opSize; ++i) {
        auto op = net->net->oplists()->GetAs<Op>(i);
        if (nullptr == op || nullptr == op->outputIndexes()) {
            MNN_ERROR("Invalid Model, the %d op is empty\n", i);
            delete net;
            return nullptr;
        }
    }
    return new Interpreter(net);
}

void Interpreter::setSessionHint(HintMode mode, int hint) {
    mNet->modes.setHint(mode, hint);
}

void Interpreter::setSessionMode(SessionMode mode) {
    if (mode == Session_Resize_Check) {
        for (auto& iter : mNet->sessions) {
            iter->openResizeCheck();
        }
    } else if (mode == Session_Resize_Fix) {
        for (auto& iter : mNet->sessions) {
            iter->fixResizeCache();
        }
    } else {
        mNet->modes.setMode(mode);
    }
}

void Interpreter::setCacheFile(const char* cacheFile, size_t keySize) {
    if (nullptr == cacheFile || nullptr == mNet->buffer.get()) {
        MNN_ERROR("Empty cacheFile or the interpreter invalid\n");
        return;
    }
    mNet->cacheFile   = std::string(cacheFile);
    std::unique_ptr<FileLoader> loader(new FileLoader(cacheFile, true));
    if (!loader->valid()) {
        MNN_ERROR("Load Cache file error.\n");
        return;
    }
    bool result = loader->read();
    if (!result) {
        MNN_ERROR("Load Cache file error.\n");
        return;
    }
    if (loader->size() == 0) {
        MNN_ERROR("Load Cache file error.\n");
        return;
    }
    bool success = loader->merge(mNet->cacheBuffer);
    if (!success) {
        MNN_ERROR("Alloc memory for Cache error.\n");
        return;
    }
}

void Interpreter::setExternalFile(const char* file, size_t flag) {
    mNet->externalFile = file;
}

ErrorCode Interpreter::updateCacheFile(Session *session, int flag) {
    std::lock_guard<std::mutex> _l(mNet->lock);

    // Backend_Auto and no Async work, then don't need updateCache
    if(mNet->modes.backendMode == Session_Backend_Auto && !(session->hasAsyncWork())) {
        return NO_ERROR;
    }
    
    // Get cache and write to file
    auto buffer = session->getCache();

    //When current cacheSize bigger than previous, update
    if (buffer.first != nullptr && buffer.second > mNet->lastCacheSize) {
        MNN_PRINT("Update cache to %s, from size:%zu -> size:%zu\n", mNet->cacheFile.c_str(), mNet->lastCacheSize, buffer.second);
        writeCacheFile(mNet, buffer);
        mNet->lastCacheSize = buffer.second;
    }
    // Reset cache
    session->loadCache(nullptr, 0);
    return NO_ERROR;
}

Interpreter::Interpreter(Content* net) {
    MNN_ASSERT(nullptr != net);
    mNet = net;
    // Store bizcode and uuid because we need them even after `releaseModel` is called.
    mNet->bizCode = std::string(mNet->net->bizCode() ? mNet->net->bizCode()->c_str() : "");
    mNet->uuid = std::string(mNet->net->mnn_uuid() ? mNet->net->mnn_uuid()->c_str() : "");
#ifdef MNN_INTERNAL_ENABLED
    mNet->basicLogginData = logBasicInfo();
    mNet->basicLogginData.emplace("ModelVersion", getModelVersion());
#endif
}

Interpreter::~Interpreter() {
    {
        // If the session is running, we must not delete session
        std::unique_lock<std::mutex> _l(mNet->lock);
        mNet->sessions.clear();
        mNet->tensorMap.clear();
    }
    delete mNet;
}

Session* Interpreter::createMultiPathSession(const std::vector<ScheduleConfig>& configs) {
    RuntimeInfo runtime = createRuntime(configs);
    if (runtime.first.empty()) {
        MNN_ERROR("Runtime not valid for create session\n");
        return nullptr;
    }
    return createMultiPathSession(configs, std::move(runtime));
}

Session* Interpreter::createMultiPathSession(const std::vector<ScheduleConfig>& configs, const RuntimeInfo& runtime) {
    for (auto& iter : runtime.first) {
        iter.second->setRuntimeHint(mNet->modes.runtimeHint);
    }
    runtime.second->setRuntimeHint(mNet->modes.runtimeHint);

    if (nullptr == mNet->buffer.get()) {
        MNN_ERROR("The model buffer has been released. Can't create session\n");
        return nullptr;
    }
    if (runtime.first.empty()) {
        MNN_ERROR("Runtime not valid for create session\n");
        return nullptr;
    }
    std::unique_lock<std::mutex> _l(mNet->lock);
#ifdef MNN_INTERNAL_ENABLED
    Timer _timer;
#endif
    int cacheMode = 0; // No cache
    Schedule::ScheduleInfo info;
    info.externalWeightPath = mNet->externalFile;
    auto success = Schedule::schedule(info, mNet->net, configs, runtime);
    if (!success) {
        return nullptr;
    }
    if (info.needInputContentForShape) {
        MNN_ERROR("Interpreter don't support case for shape compute need input content, please use module api instead\n");
        return nullptr;
    }
    RuntimeInfo rt = runtime;
    bool valid  = false;
    if (mNet->cacheBuffer.get() != nullptr) {
        for (auto iter : rt.first) {
            valid = iter.second->onSetCache(mNet->cacheBuffer.get(),
                                            mNet->cacheBuffer.size());
            if(!valid) {
                iter.second->onSetCache(nullptr, 0);
            }
            if (valid) {
                break;
            }
        }
        if (valid) {
            mNet->lastCacheSize = mNet->cacheBuffer.size();
            cacheMode = cacheMode | 1; // READ cache
        }
    }

    auto newSession =
        std::unique_ptr<Session>(new Session(std::move(info), mNet->modes, std::move(rt)));
    if (!newSession->valid()) {
        MNN_PRINT("Invalide Session!!\n");
        return nullptr;
    }
    auto result = newSession.get();
    auto validForResize = info.validForResize;
    if (validForResize && mNet->modes.inputMode == Session_Input_Inside && mNet->modes.resizeMode == Session_Resize_Direct) {
        result->resize();
    }

    if ((!mNet->cacheFile.empty()) && (!valid) && mNet->modes.backendMode == Session_Backend_Fix) {
        // Try to save extra cache
        auto buffer = result->getCache();
        if (buffer.first != nullptr && buffer.second > 0) {
            MNN_PRINT("Write cache to %s, size = %zu\n", mNet->cacheFile.c_str(), buffer.second);
            writeCacheFile(mNet, buffer);
            mNet->lastCacheSize = buffer.second;
            // Write Cache
            cacheMode = cacheMode | 2;
        }
    }
    // Reset cache
    result->loadCache(nullptr, 0);

    mNet->sessions.emplace_back(std::move(newSession));

#ifdef MNN_INTERNAL_ENABLED
    int precision = BackendConfig::Precision_Normal;
    if (nullptr != configs[0].backendConfig) {
        precision = configs[0].backendConfig->precision;
    }
    int mode = configs[0].mode;
    mNet->sessionInfo.insert(std::make_pair(result, std::make_tuple(precision, mode)));
    if (shouldLog(FREQ_HIGH)) {
        std::map<std::string, std::string> metrics = mNet->basicLogginData;
        metrics.emplace("UUID", mNet->uuid);
        metrics.emplace("Time", std::to_string((float)_timer.durationInUs() / 1024.0f));
        metrics.emplace("Backend", std::to_string(configs[0].type));
        metrics.emplace("Precision", std::to_string(precision));
        metrics.emplace("Mode", std::to_string(mode));
        metrics.emplace("Cache", std::to_string(cacheMode));
        metrics.emplace("CacheSize", std::to_string((float)(mNet->lastCacheSize / 1024.0f)));
        metrics.emplace("ModelSize", std::to_string ((float)mNet->buffer.size() / 1024.0f / 1024.0f));
        metrics.emplace("Usage", std::to_string((int) mNet->net->usage()));
        metrics.emplace("API", "Interpreter::createMultiPathSession");
        logAsync(metrics);
    }
#endif // MNN_INTERNAL_ENABLED

    return result;
}

Session* Interpreter::createSession(const ScheduleConfig& config) {
    return createMultiPathSession({config});
}

Session* Interpreter::createSession(const ScheduleConfig& config, const RuntimeInfo& runtime) {
    return createMultiPathSession({config}, runtime);
}

bool Interpreter::releaseSession(Session* session) {
    std::unique_lock<std::mutex> _l(mNet->lock);
    for (auto iter = mNet->sessions.begin(); iter != mNet->sessions.end(); iter++) {
        // TODO Delete tensormap
        for (auto tIter = mNet->tensorMap.begin(); tIter != mNet->tensorMap.end();) {
            if (tIter->second == session) {
                tIter = mNet->tensorMap.erase(tIter);
                continue;
            }
            tIter++;
        }

        if ((*iter).get() == session) {
            mNet->sessions.erase(iter);
            return true;
        }
    }
    return false;
}

#ifdef MNN_INTERNAL_ENABLED
void Interpreter::logForRunSession(const Session* session, float timeInMs, const char* api) const {
    int backendType[MNN_FORWARD_ALL] ;
    session->getInfo(MNN::Interpreter::BACKENDS, backendType);
    float flops = 0.0f;
    session->getInfo(MNN::Interpreter::FLOPS, &flops);
    float memory = 0.0f;
    session->getInfo(MNN::Interpreter::MEMORY, &memory);
    std::map<std::string, std::string> metrics = mNet->basicLogginData;
    metrics.emplace("UUID", mNet->uuid);
    metrics.emplace("Backend", std::to_string(backendType[0])); // "Precision" is not logged here. Don't need it.
    metrics.emplace("Time", std::to_string(timeInMs));
    metrics.emplace("API", api);
    metrics.emplace("Flops", std::to_string(flops));
    metrics.emplace("Memory", std::to_string(memory));
    auto iter = mNet->sessionInfo.find(session);
    if (iter != mNet->sessionInfo.end()) {
        metrics.emplace("Precision", std::to_string(std::get<0>(iter->second)));
        metrics.emplace("Mode", std::to_string(std::get<1>(iter->second)));
    }
    logAsync(metrics);
}
#endif

ErrorCode Interpreter::runSession(Session* session) const {
    std::unique_lock<std::mutex> _l(mNet->lock);
#ifdef MNN_INTERNAL_ENABLED
    Timer timer;
#endif
    ErrorCode errorcode = session->run();

#ifdef MNN_INTERNAL_ENABLED
    if (shouldLog(FREQ_LOW)) {
        waitSessionFinish(session);
        float costTime = (float)timer.durationInUs() / (float)1000;
        logForRunSession(session, costTime, "Interpreter::runSession");
    }
#endif // MNN_INTERNAL_ENABLED

    return errorcode;
}

Tensor* Interpreter::getSessionInput(const Session* session, const char* name) {
    if (session == nullptr) {
        return nullptr;
    }
    std::unique_lock<std::mutex> _l(mNet->lock);
    auto tensor = session->getInput(name);
    mNet->tensorMap.insert(std::make_pair(tensor, session));
    return tensor;
}

Tensor* Interpreter::getSessionOutput(const Session* session, const char* name) {
    if (session == nullptr) {
        return nullptr;
    }
    std::unique_lock<std::mutex> _l(mNet->lock);
    auto tensor = session->getOutput(name);
    mNet->tensorMap.insert(std::make_pair(tensor, session));
    return tensor;
}

const std::map<std::string, Tensor*>& Interpreter::getSessionInputAll(const Session* session) const {
    std::unique_lock<std::mutex> _l(mNet->lock);
    auto& tensors = session->getInputAll();
    for (auto& iter : tensors) {
        mNet->tensorMap.insert(std::make_pair(iter.second, session));
    }
    return tensors;
}

const std::map<std::string, Tensor*>& Interpreter::getSessionOutputAll(const Session* session) const {
    std::unique_lock<std::mutex> _l(mNet->lock);
    auto& tensors = session->getOutputAll();
    for (auto& iter : tensors) {
        mNet->tensorMap.insert(std::make_pair(iter.second, session));
    }
    return tensors;
}
void Interpreter::resizeSession(Session* session) {
    resizeSession(session, 0);
}

void Interpreter::resizeSession(Session* session, int needRelloc) {
    std::unique_lock<std::mutex> _l(mNet->lock);
    if (mNet->buffer.get() == nullptr) {
        MNN_ERROR("The model buffer has been released. Can't resize session\n");
        return;
    }
    if (1 == needRelloc) {
        session->setNeedMalloc(true);
    }
    session->resize();
}

ErrorCode Interpreter::runSessionWithCallBack(const Session* session, const TensorCallBack& before,
                                              const TensorCallBack& after, bool sync) const {
    auto beforeWrap = [&before](const std::vector<Tensor*>& tensors, const OperatorInfo* info) {
        return before(tensors, info->name());
    };
    auto afterWrap = [&after](const std::vector<Tensor*>& tensors, const OperatorInfo* info) {
        return after(tensors, info->name());
    };
    return runSessionWithCallBackInfo(session, beforeWrap, afterWrap, sync);
}

void Interpreter::waitSessionFinish(const Session* session) const {
    for (auto& t : mNet->tensorMap) {
        if (t.second == session) {
            if (TensorUtils::getDescribe(t.first)->usage != Tensor::InsideDescribe::INPUT) {
                t.first->wait(Tensor::MAP_TENSOR_READ, true);
            }
        }
    }
}

ErrorCode Interpreter::runSessionWithCallBackInfo(const Session* session, const TensorCallBackWithInfo& before,
                                                  const TensorCallBackWithInfo& callBack, bool sync) const {

    std::unique_lock<std::mutex> _l(mNet->lock);
#ifdef MNN_INTERNAL_ENABLED
    Timer timer;
#endif
    ErrorCode errorcode = session->runWithCallBack(before, callBack, sync);

#ifdef MNN_INTERNAL_ENABLED
    if (shouldLog(FREQ_LOW)) {
        waitSessionFinish(session);
        float costTime = (float)timer.durationInUs() / (float)1000;
        logForRunSession(session, costTime, "Interpreter::runSessionWithCallBackInfo");
    }
#endif // MNN_INTERNAL_ENABLED

    return errorcode;
}

const Backend* Interpreter::getBackend(const Session* session, const Tensor* tensor) const {
    return session->getBackEnd(tensor);
}

void Interpreter::releaseModel() {
    std::unique_lock<std::mutex> _l(mNet->lock);
    for (auto& session : mNet->sessions) {
        session->waitAsyncResize();
    }
    if (mNet->buffer.get() != nullptr && mNet->net->usage() != Usage_INFERENCE_STATIC) {
        mNet->buffer.release();
    }
    mNet->cacheBuffer.release();
}

void Interpreter::resizeTensor(Tensor* tensor, int batch, int channel, int height, int width) {
    if (tensor->getDimensionType() == Tensor::TENSORFLOW) {
        resizeTensor(tensor, {batch, height, width, channel});
    } else {
        resizeTensor(tensor, {batch, channel, height, width});
    }
}

void Interpreter::resizeTensor(Tensor* tensor, const std::vector<int>& dims) {
    std::unique_lock<std::mutex> _l(mNet->lock);
    MNN_ASSERT(nullptr != tensor);
    bool dirty = false;
    if (tensor->buffer().dimensions != dims.size()) {
        dirty = true;
    } else {
        for (int i = 0; i < dims.size(); ++i) {
            if (tensor->buffer().dim[i].extent != dims[i]) {
                dirty = true;
                break;
            }
        }
    }

    if (!dirty) {
        return;
    }

    tensor->buffer().dimensions = (int)dims.size();
    for (int i = 0; i < dims.size(); ++i) {
        tensor->buffer().dim[i].extent = dims[i];
    }

    auto relatedSessionIter = mNet->tensorMap.find(tensor);
    MNN_ASSERT(relatedSessionIter != mNet->tensorMap.end());
    ((MNN::Session*)relatedSessionIter->second)->setNeedResize();
}

const char* Interpreter::bizCode() const {
    return mNet->bizCode.c_str();
}

const char* Interpreter::uuid() const {
    return mNet->uuid.c_str();
}

std::pair<const void*, size_t> Interpreter::getModelBuffer() const {
    return std::make_pair(mNet->buffer.get(), mNet->buffer.size());
}
ErrorCode Interpreter::updateSessionToModel(Session* session) {
    std::unique_lock<std::mutex> _l(mNet->lock);
    if (mNet->buffer.get() == nullptr) {
        MNN_ERROR("Can't updateSessionToModel because you called releaseModel before\n");
        return INPUT_DATA_ERROR;
    }
    return session->updateToModel((Net*)mNet->net);
}

const char* Interpreter::getModelVersion() const {
    if (mNet && mNet->net && mNet->net->extraInfo() && mNet->net->extraInfo()->version()) {
        return mNet->net->extraInfo()->version()->c_str();
    }
    return "<2.0.0";
}

bool Interpreter::getSessionInfo(const Session* session, SessionInfoCode code, void* ptr) {
    std::unique_lock<std::mutex> _l(mNet->lock);
    if (nullptr == session || nullptr == ptr) {
        return false;
    }
    return session->getInfo(code, ptr);
}

static void _getDefaultBackend(RuntimeInfo& rt) {
    auto defaultType = MNN_FORWARD_CPU;
    if (rt.first.find(defaultType) != rt.first.end()) {
        rt.second = rt.first[defaultType];
    }
    if (rt.second == nullptr) {
        Backend::Info info;
        info.type      = defaultType;
        info.numThread = 1;
        rt.second.reset(RuntimeFactory::create(info));
    }
}
RuntimeInfo Interpreter::createRuntime(const std::vector<ScheduleConfig>& configs) {
    RuntimeInfo res;
    auto& mRuntimes = res.first;
    for (auto& config : configs) {
        Backend::Info compute;
        compute.type      = Schedule::getApprociateType(config);
        compute.numThread = config.numThread;
        if(config.type == MNN_FORWARD_AUTO) {
            if(compute.type == MNN_FORWARD_OPENCL || compute.type == MNN_FORWARD_METAL) {
                // AUTO set default gpu-mode MNN_GPU_TUNING_FAST
                compute.numThread = 16;
            }
        }
        compute.user      = config.backendConfig;
        if (mRuntimes.find(compute.type) == mRuntimes.end()) {
            auto newBn = RuntimeFactory::create(compute);
            if (nullptr == newBn) {
                MNN_ERROR("Can't create Runtime: %s\n", EnumNameForwardType((ForwardType)compute.type));
                continue;
            }
            mRuntimes[compute.type].reset(newBn);
        }
    }
    _getDefaultBackend(res);
    return res;
}
void Interpreter::destroy(Interpreter* net) {
    if (nullptr != net) {
        delete net;
    }
}

} // namespace MNN
