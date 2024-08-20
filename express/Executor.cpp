//
//  Executor.cpp
//  MNN
//
//  Created by MNN on 2019/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Executor.hpp>
#include "core/TensorUtils.hpp"
#include "core/FileLoader.hpp"
#include "Utils.hpp"
#include <MNN/AutoTime.hpp>
#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include <MNN/expr/ExecutorScope.hpp>
#include "core/Backend.hpp"
#include "RuntimeAttr.hpp"
#include <stack>
#define DEFAULT_BACKUP_RUNTIME_KEY MNN_FORWARD_CPU
#ifdef MNN_EXPR_ENABLE_PROFILER
#define MNN_EXPRESS_ERROR_REPORT
#endif

namespace MNN {
namespace Express {

void Executor::setGlobalExecutorConfig(MNNForwardType type, const BackendConfig& config, int numberThread) {
    std::lock_guard<std::mutex> _l(mMutex);
    
    if(type == MNN_FORWARD_AUTO) {
        ScheduleConfig sConfig;
        sConfig.type = type;
        type = Schedule::getApprociateType(sConfig);
        auto creator = MNNGetExtraRuntimeCreator(type);
        MNN_ASSERT(nullptr != creator);
        Backend::Info info;
        info.type = type;
        info.mode = Backend::Info::DIRECT;
        info.numThread = numberThread;
        if(type == MNN_FORWARD_OPENCL || type == MNN_FORWARD_METAL) {
            info.numThread = 4;
        }
        mAttr->firstType = type;
        auto firstIter = mRuntimes.find(mAttr->firstType);
        if (firstIter == mRuntimes.end()) {
            info.user = (BackendConfig*)&config;
            std::shared_ptr<Runtime> bn(creator->onCreate(info));
            mRuntimes[mAttr->firstType] = bn;
        } else {
            firstIter->second->onReset(numberThread, &config);
        }
    } else {
        auto creator = MNNGetExtraRuntimeCreator(type);
        if (nullptr == creator) {
            MNN_ERROR("Error to find creator of %d, set CPU default\n", type);
            type = MNN_FORWARD_CPU;
            creator = MNNGetExtraRuntimeCreator(type);
        }
        MNN_ASSERT(nullptr != creator);
        Backend::Info info;
        info.type = type;
        mAttr->firstType = type;
        auto firstIter = mRuntimes.find(mAttr->firstType);
        if (firstIter == mRuntimes.end()) {
            info.mode = Backend::Info::DIRECT;
            info.numThread = numberThread;
            info.user = (BackendConfig*)&config;
            std::shared_ptr<Runtime> bn(creator->onCreate(info));
            mRuntimes[mAttr->firstType] = bn;
        } else {
            firstIter->second->onReset(numberThread, &config);
        }
    }
    _refreshRuntime();
}

int Executor::getCurrentRuntimeStatus(RuntimeStatus statusEnum) {
    return mRuntimes[mAttr->firstType]->onGetRuntimeStatus(statusEnum);
}

void Executor::gc(GCFlag flag) {
    int level = flag == FULL ? 100 : 0;
    for (auto& iter : mRuntimes) {
        iter.second->onGabageCollect(level);
    }
}

Executor::Executor(std::shared_ptr<Runtime> backend, MNNForwardType type, int numberThread) {
    mRuntimes.insert(std::make_pair(type, backend));
    mAttr.reset(new ExecutorAttr);
    mAttr->firstType = type;
    if (MNN_FORWARD_CPU != type) {
        // Create Backup Backend
        Backend::Info info;
        info.type = MNN_FORWARD_CPU;
        auto cre = MNNGetExtraRuntimeCreator(MNN_FORWARD_CPU);
        info.mode = Backend::Info::DIRECT;
        info.numThread = 1;
        std::shared_ptr<Runtime> backupRt(cre->onCreate(info));
        mRuntimes.insert(std::make_pair(DEFAULT_BACKUP_RUNTIME_KEY, backupRt));
    }
    mDebug.reset(new DebugTools);
    BackendConfig defaultConfig;
    defaultConfig.flags = 4;
    std::shared_ptr<Backend> defaultBackend(mRuntimes[DEFAULT_BACKUP_RUNTIME_KEY]->onCreate(&defaultConfig));
    mAttr->constantBackend = defaultBackend;
    _refreshRuntime();
}
Executor::~Executor(){
    // Do nothing
}

void Executor::setCallBack(TensorCallBackWithInfo&& before, TensorCallBackWithInfo&& after) {
    mDebug->before = std::move(before);
    mDebug->after = std::move(after);
}

Executor::Requirement Executor::getRequirement(Expr* expr) const {
    Executor::Requirement req;
    auto op = expr->get();
    auto inputSize = expr->inputs().size();
    req.contentNeedContent.resize(inputSize);
    req.shapeNeedContent.resize(inputSize);
    if (op->type() == OpType_Extra) {
        for (int i = 0; i < inputSize; ++i) {
            req.contentNeedContent[i] = true;
            req.shapeNeedContent[i]   = false;
        }
        return req;
    }
    for (int i = 0; i < inputSize; ++i) {
        req.contentNeedContent[i] = OpCommonUtils::opNeedContent(op, i);
        req.shapeNeedContent[i]   = false;
    }
    auto needIndexId = SizeComputer::needInputContent(op, inputSize);
    for (auto index : needIndexId) {
        if (index < req.shapeNeedContent.size()) {
            req.shapeNeedContent[index] = true;
        }
    }
    return req;
}

static std::once_flag gInitFlag;
static std::shared_ptr<Executor>* gExecutor = nullptr;
std::shared_ptr<Executor> Executor::getGlobalExecutor() {
    std::call_once(gInitFlag, [&]() {
        auto creator = MNNGetExtraRuntimeCreator(MNN_FORWARD_CPU);
#ifdef MNN_BUILD_MINI
        SizeComputerSuite::init();
        GeometryComputer::init();
#endif
        Backend::Info info;
        info.type = MNN_FORWARD_CPU;
        info.numThread = 1;
        std::shared_ptr<Runtime> bn(creator->onCreate(info));
        RuntimeHint hint;
        hint.memoryAllocatorType = 0;// Defer
        bn->setRuntimeHint(hint);
        gExecutor = new std::shared_ptr<Executor>(new Executor(bn, MNN_FORWARD_CPU, 1));
    });
    return *gExecutor;
}

std::shared_ptr<Executor> Executor::newExecutor(MNNForwardType type,
                                                const BackendConfig& config,
                                                int numberThread) {
    auto creator = MNNGetExtraRuntimeCreator(type);
    if(nullptr == creator) {
        MNN_ERROR("Don't support %d\n", type);
        return nullptr;
    }
    Backend::Info info;
    info.type = type;
    info.numThread = numberThread;
    info.user = const_cast<BackendConfig*>(&config);
    std::shared_ptr<Runtime> runtime(creator->onCreate(info));
    auto executor = new Executor(runtime, type, numberThread);
    return std::shared_ptr<Executor>(executor);
}
void Executor::_refreshRuntime() {
    mRuntimeInfo.first.clear();
    mRuntimeInfo.second = mRuntimes[DEFAULT_BACKUP_RUNTIME_KEY];
    auto firstIter = mRuntimes.find(getAttr()->firstType);
    if (firstIter != mRuntimes.end()) {
        mRuntimeInfo.first.insert(std::make_pair(firstIter->first, firstIter->second));
    } else {
        MNN_ASSERT(false);
    }
    for (auto& iter : mRuntimes) {
        if (iter.first != getAttr()->firstType) {
            mRuntimeInfo.first.insert(std::make_pair(iter.first, iter.second));
        }
    }
}

RuntimeInfo Executor::getRuntime() {
    auto glo = ExecutorScope::Current();
    return glo->mRuntimeInfo;
}
bool Executor::getComputeInfo(EXPRP expr, Interpreter::SessionInfoCode code, void* ptr) {
    if (nullptr == expr) {
        return false;
    }
    if (nullptr == expr->inside()->mCache.get()) {
        return false;
    }
    auto session = expr->inside()->mCache->getSession();
    if (nullptr == session) {
        return false;
    }
    return session->getInfo(code, ptr);
}

static bool loadCache(std::shared_ptr<Runtime> &rt, const void* buffer, size_t size) {
    auto res = rt->onSetCache(buffer, size);
    if (res) {
        return true;
    }
    return false;
}
static std::pair<const void*, size_t> getCache(std::shared_ptr<Runtime> &rt) {
    auto res = rt->onGetCache();
    if (res.first != nullptr) {
        return res;
    }
    return std::make_pair(nullptr, 0);
}

static void writeCacheFile(std::shared_ptr<Cache> cache, std::pair<const void*, size_t> buffer) {
    auto verifyInfo = std::make_pair((const void*)cache->modelBuffer.get(), cache->cacheOffset);
    bool res = FileLoader::write(cache->cacheFile.c_str(), buffer);
    if (!res) {
        MNN_ERROR("Write Cache File error!\n");
        return;
    }
}
Executor::RuntimeManager* Executor::RuntimeManager::createRuntimeManager(std::vector<ScheduleConfig>& configs) {
    if (configs.empty()) {
        return nullptr;
    }
    return createRuntimeManager(configs[0]);
}
void Executor::RuntimeManager::destroy(RuntimeManager* rtmgr) {
    if (nullptr != rtmgr) {
        delete rtmgr;
    }
}

void Executor::RuntimeManager::setMode(Interpreter::SessionMode mode) {
    mInside->modes.setMode(mode);
}
void Executor::RuntimeManager::setHint(Interpreter::HintMode mode, int value) {
    mInside->modes.setHint(mode, value);
}
bool Executor::RuntimeManager::getInfo(Interpreter::SessionInfoCode code, void* ptr) {
    // Only support get memory
    switch (code) {
        case Interpreter::MEMORY: {
            auto dst     = (float*)ptr;
            float summer = mInside->mRuntime.second->onGetMemoryInMB();
            for (auto& r : mInside->mRuntime.first) {
                if (r.second.get() != mInside->mRuntime.second.get()) {
                    summer += r.second->onGetMemoryInMB();
                }
            }
            *dst = summer;
            return true;
        } break;
        case Interpreter::BACKENDS: {
            auto dst = (int*)ptr;
            if (!mInside->mRuntime.first.empty()) {
                *dst = mInside->mRuntime.first.begin()->first;
            }
        } break;
        default: {
            // Do nothing
        } break;
    }
    return false;
}

Executor::RuntimeManager::RuntimeManager() {
    mInside = new RuntimeAttr;
    // Default set release for better performance
    mInside->modes.callBackMode = Interpreter::Session_Release;
    mInside->modes.inputMode = Interpreter::Session_Input_User;
    mInside->modes.outputMode = Interpreter::Session_Output_User;
}
Executor::RuntimeManager::~RuntimeManager() {
    updateCache();
    delete mInside;
}
Executor::RuntimeManager* Executor::RuntimeManager::createRuntimeManager(const ScheduleConfig &config) {
    auto res = new RuntimeManager;
    auto glo = ExecutorScope::Current();
    std::lock_guard<std::mutex> _l(glo->mMutex);
    auto& originRt = glo->mRuntimes;
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
    auto iter = originRt.find(compute.type);
    if (iter == originRt.end()) {
        auto creator = MNNGetExtraRuntimeCreator(compute.type);
        if (nullptr == creator) {
            return nullptr;
        }
        auto newBn = creator->onCreate(compute);
        if (nullptr == newBn) {
            MNN_ERROR("Can't create Runtime: %s\n", EnumNameForwardType((ForwardType)compute.type));
            return nullptr;
        }
        originRt.insert(std::make_pair(compute.type, std::shared_ptr<Runtime>(newBn)));
    } else {
        iter->second->onReset(compute.numThread, compute.user);
    }
    res->mInside->mRuntime.second =  originRt[DEFAULT_BACKUP_RUNTIME_KEY];
    res->mInside->mRuntime.first.insert(std::make_pair(compute.type, originRt[compute.type]));
    res->mInside->mInfo = originRt[compute.type];
    res->mInside->mNumberThread = compute.numThread;
    if (nullptr != config.backendConfig) {
        res->mInside->mConfig = *config.backendConfig;
        res->mInside->mUserConfig = true;
    } else {
        res->mInside->mUserConfig = false;
    }
    glo->_refreshRuntime();
    return res;
}
ExecutorAttr* Executor::getAttr() const {
    return mAttr.get();
}

BackendConfig* Executor::RuntimeManager::getBnConfig() {
    if (mInside->mUserConfig) {
        return &mInside->mConfig;
    }
    return nullptr;
}


void Executor::RuntimeManager::setCache(std::string cacheName) {
    std::lock_guard<std::mutex> _l(mLock);

    mInside->mCache.reset(new Cache);
    mInside->mCache->cacheFile = cacheName;
    if (nullptr == mInside->mCache->cacheFile.c_str()) {
        MNN_ERROR("Empty cacheFile\n");
        return;
    }
    std::unique_ptr<FileLoader> loader(new FileLoader(mInside->mCache->cacheFile.c_str(), true));
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
    bool success = loader->merge(mInside->mCache->cacheBuffer);
    if (!success) {
        MNN_ERROR("Alloc memory for Cache error.\n");
        return;
    }

    // load cache
    bool valid = loadCache(mInside->mInfo, mInside->mCache->cacheBuffer.get() + mInside->mCache->cacheOffset,
                           mInside->mCache->cacheBuffer.size() - mInside->mCache->cacheOffset);
    if(!valid) {
        // Reset cache
        loadCache(mInside->mInfo, nullptr, 0);
        MNN_PRINT("Cache invalid, will be reset\n");
    } else {
        mInside->mCache->lastCacheSize = mInside->mCache->cacheBuffer.size() - mInside->mCache->cacheOffset;
    }
}

void Executor::RuntimeManager::setExternalFile(std::string fileName) {
    mInside->mExternalFile = fileName;
}

void Executor::RuntimeManager::updateCache() {
    if (nullptr == mInside->mCache) {
        return;
    }
    std::lock_guard<std::mutex> _l(mLock);

    // Backend_Auto and no Async work, then don't need updateCache
    if(mInside->modes.backendMode == Interpreter::Session_Backend_Auto && !(mInside->mInfo->hasAsyncWork())) {
        return;
    }

    // Set mCancelled for quickly ending
    mInside->mInfo->mCancelled = true;
    mInside->mInfo->waitAsyncWork();
    auto buffer = getCache(mInside->mInfo);

    //When current cacheSize bigger than previous, update
    if (buffer.first != nullptr && buffer.second > mInside->mCache->lastCacheSize) {
        MNN_PRINT("Update cache to %s, size = %zu\n", mInside->mCache->cacheFile.c_str(), buffer.second);
        writeCacheFile(mInside->mCache, buffer);
        mInside->mCache->lastCacheSize = buffer.second;
    }
    // Reset cache
    loadCache(mInside->mInfo, nullptr, 0);
}

std::vector<bool> Executor::RuntimeManager::isBackendSupport(const std::vector<MNNForwardType> types) {
    std::vector<bool> res;
    for (auto bn : types) {
        auto rt = MNNGetExtraRuntimeCreator(bn);
        if (rt != nullptr) {
            res.push_back(true);
        } else {
            res.push_back(false);
        }
    }
    return res;
}

ErrorCode Executor::computeInfo(Expr* expr) {
    MNN_ASSERT(nullptr != expr);
    MNN_ASSERT(nullptr != expr->get());
    if (expr->get()->type() == OpType_Extra) {
        return NOT_SUPPORT;
    }
    auto op = expr->get();
    std::vector<Tensor*> inputTensors(expr->inputs().size());
    for (int i=0; i<inputTensors.size(); ++i) {
        auto inputExpr = expr->inputs()[i]->expr();
        inputTensors[i] = inputExpr.first->inside()->mOutputTensors[inputExpr.second];
    }
    bool res = SizeComputer::computeOutputSize(op, inputTensors, expr->inside()->mOutputTensors);
    if (!res) {
        // Compute Error
#ifdef MNN_EXPRESS_ERROR_REPORT
        if (expr->name().empty()) {
            MNN_ERROR("Error to compute shape for %s\n", EnumNameOpType(op->type()));
        } else {
            MNN_ERROR("Error to compute shape for %s, %s\n", EnumNameOpType(op->type()), expr->name().c_str());
        }
#endif
        return COMPUTE_SIZE_ERROR;
    }
    for (int i = 0; i < expr->outputSize(); ++i) {
        auto tensor = expr->inside()->mOutputTensors[i];
        TensorUtils::setLinearLayout(tensor);
        auto shape  = expr->outputInfo(i);
        Utils::copyTensorToInfo(shape, tensor);
    }
    return NO_ERROR;
}

void Executor::_makeCache(const std::vector<EXPRP>& expr, bool forceCPU) {
    std::set<std::shared_ptr<Executor::ComputeCache>> inputCaches;
    std::set<std::shared_ptr<Expr::Inside>> inputNode;
    std::stack<EXPRP> dfsStack;
    // first: target expr, second: tensor offset
    std::map<EXPRP, int> dstExpr;
    std::map<EXPRP, int> visited;
    std::set<std::shared_ptr<Expr::Inside>> extraInputs;
    for (auto e : expr) {
        if (e->get() != nullptr) {
            dfsStack.push(e);
            dstExpr.insert(std::make_pair(e, -1));
        }
    }
    if (dfsStack.empty()) {
        return;
    }
    auto current = ExecutorScope::Current();
    auto rt = current->getRuntime();
    Schedule::ScheduleInfo scheduleInfo;
    scheduleInfo.externalWeightPath = current->getAttr()->externalFile;
    scheduleInfo.pipelineInfo.resize(1);
    auto& pipeline = scheduleInfo.pipelineInfo[0].second;
    std::vector<std::shared_ptr<BufferStorage>> opBuffers;
    while (!dfsStack.empty()) {
        auto expr = dfsStack.top();
        auto& inputs = expr->inputs();
        auto& req = expr->inside()->mReq.contentNeedContent;
        MNN_ASSERT(inputs.size() == req.size());
        bool ready = true;
        for (int i = 0; i < inputs.size(); ++i) {
            if (!req[i]) {
                continue;
            }
            auto inputExpr = inputs[i]->expr();
            if (nullptr == inputExpr.first->get()) {
                if (VARP::INPUT == inputExpr.first->inputType()) {
                    extraInputs.insert(inputExpr.first->inside());
                }
                continue;
            }
            auto inputCache = inputExpr.first->inside()->mCache;
            if (nullptr != inputCache) {
                inputCaches.insert(inputCache);
                continue;
            }
            if (visited.find(inputExpr.first) != visited.end()) {
                continue;
            }
            ready = false;
            dfsStack.push(inputExpr.first);
            break;
        }
        if (!ready) {
            continue;
        }
        dfsStack.pop();
        int currentIndex = (int)pipeline.size();
        visited.insert(std::make_pair(expr, currentIndex));
        Schedule::OpCacheInfo opInfo;
        opInfo.op = expr->get();
        opBuffers.emplace_back(expr->extra());
        opInfo.inputs.resize(inputs.size());
        opInfo.outputs.resize(expr->outputSize());
        int offset = scheduleInfo.allTensors.size();
        for (int i=0; i<opInfo.outputs.size(); ++i) {
            std::shared_ptr<Tensor> tensor(new Tensor);
            opInfo.outputs[i] = tensor.get();
            auto srcTensor = expr->inside()->mOutputTensors[i];
            TensorUtils::copyShape(srcTensor, tensor.get(), true, true);
            if (TensorUtils::getDescribe(srcTensor)->quantAttr.get()) {
                TensorUtils::getDescribe(tensor.get())->quantAttr.reset(new QuantAttr);
                auto quant = TensorUtils::getDescribe(tensor.get())->quantAttr.get();
                quant->scale = TensorUtils::getDescribe(srcTensor)->quantAttr.get()->scale;
                quant->zero = TensorUtils::getDescribe(srcTensor)->quantAttr.get()->zero;
            }

            TensorUtils::getDescribe(tensor.get())->index = (int)scheduleInfo.allTensors.size();
            scheduleInfo.allTensors.emplace_back(tensor);
        }
        auto dstIter = dstExpr.find(expr);
        if (dstIter != dstExpr.end()) {
            dstIter->second = offset;
            for (int i=0; i<opInfo.outputs.size(); ++i) {
                TensorUtils::getDescribe(opInfo.outputs[i])->usage = Tensor::InsideDescribe::OUTPUT;
            }
        }
        for (int i = 0; i < inputs.size(); ++i) {
            auto inputExpr = inputs[i]->expr();
            if (!req[i]) {
                opInfo.inputs[i] = Utils::getTensor(inputs[i]);
                continue;
            }
            if (nullptr == inputExpr.first->get()) {
                opInfo.inputs[i] = Utils::getTensor(inputs[i]);
                continue;
            }
            auto inputCache = inputExpr.first->inside()->mCache;
            if (nullptr != inputCache) {
                opInfo.inputs[i] = opInfo.inputs[i] = Utils::getTensor(inputs[i]);
                continue;
            }
            auto iter = visited.find(inputExpr.first);
            MNN_ASSERT(iter != visited.end());
            opInfo.inputs[i] = pipeline[iter->second].outputs[inputExpr.second];
        }
        pipeline.emplace_back(std::move(opInfo));
    }
    Session::ModeGroup group;
    group.inputMode = Interpreter::Session_Input_User;
    group.outputMode = Interpreter::Session_Output_User;
    auto globalExecutor = ExecutorScope::Current();
    auto debug = globalExecutor->getDebugTools();
    if (debug->after != nullptr && debug->before != nullptr) {
        group.callBackMode = Interpreter::Session_Debug;
    } else {
        group.callBackMode = Interpreter::Session_Release;
    }
    group.memoryUsageMode = Interpreter::Session_Memory_Cache;
    std::shared_ptr<ComputeCache> cahce(new ComputeCache);
    for (auto& iter : dstExpr) {
        auto expr = iter.first;
        expr->inside()->mCacheOffset = iter.second;
        expr->inside()->mCache = cahce;
    }
    cahce->mCacheBuffers = std::move(opBuffers);
    // Don't report error when use expr dynamic compute, which will be called in model convert
    scheduleInfo.pipelineInfo[0].first.reportError = false;
    if (forceCPU) {
        scheduleInfo.pipelineInfo[0].first.info.type = MNN_FORWARD_CPU;
    } else {
        scheduleInfo.pipelineInfo[0].first.info.type = current->getAttr()->firstType;
    }
    scheduleInfo.pipelineInfo[0].first.needComputeShape = false;
    scheduleInfo.pipelineInfo[0].first.needComputeGeometry = mLazyMode != LAZY_CONTENT;
    cahce->mSession.reset(new Session(std::move(scheduleInfo), group, std::move(rt)));
    cahce->mInputs = inputCaches;
    cahce->mInputInside = std::move(extraInputs);
}

void Executor::makeCache(const std::vector<EXPRP>& expr, bool forceCPU) {
    //FUNC_PRINT(mCaches.size());
    _makeCache(expr, forceCPU);
}

void Executor::resetProfile() {
    // Depercated
}
void Executor::dumpProfile() {
    // Depercated
}

bool Executor::registerSubGraph(const std::string& submoduleName, VARPS outputs, VARPS inputs) {
    if (mSubGraph.find(submoduleName) != mSubGraph.end()) {
        MNN_PRINT("Executor Error: Subgraph has exists: %s\n", submoduleName.c_str());
        return false;
    }
    std::shared_ptr<SubGraph> graph(new SubGraph);
    std::vector<std::string> subInputs(inputs.size());
    std::vector<std::string> subOutputs(outputs.size());
    for (int i=0; i<inputs.size(); ++i) {
        if (inputs[i]->name().empty()) {
            MNN_PRINT("Executor Error: input %d name empty\n", i);
            return false;
        }
        subInputs[i] = inputs[i]->name();
    }
    for (int i=0; i<outputs.size(); ++i) {
        if (outputs[i]->name().empty()) {
            MNN_PRINT("Executor Error: output %d name empty\n", i);
            return false;
        }
        subOutputs[i] = outputs[i]->name();
    }
    std::unique_ptr<MNN::SubGraphProtoT> subInfo(new MNN::SubGraphProtoT);
    subInfo->name = submoduleName;
    std::unique_ptr<MNN::NetT> subNet(new MNN::NetT);
    std::vector<MNN::Express::VARP> combine = inputs;
    combine.insert(combine.end(), outputs.begin(), outputs.end());
    Variable::save(combine, subNet.get());
    std::map<std::string, int> subTensorMap;
    for (int i=0; i<subNet->tensorName.size(); ++i) {
        subTensorMap.insert(std::make_pair(subNet->tensorName[i], i));
    }
    subInfo->tensors = std::move(subNet->tensorName);
    subInfo->inputs.resize(inputs.size());
    for (int i=0; i<inputs.size(); ++i) {
        subInfo->inputs[i] = subTensorMap[subInputs[i]];
    }
    subInfo->outputs.resize(outputs.size());
    for (int i=0; i<outputs.size(); ++i) {
        subInfo->outputs[i] = subTensorMap[subOutputs[i]];
    }
    subInfo->nodes = std::move(subNet->oplists);
    for (int i=0; i<subNet->subgraphs.size(); ++i) {
        graph->depends.emplace_back(subNet->subgraphs[i]->name);
    }
    graph->info = std::move(subInfo);
    mSubGraph.insert(std::make_pair(submoduleName, graph));
    return true;
}

std::shared_ptr<Executor::SubGraph> Executor::findSubGraph(const std::string& submoduleName) {
    auto iter = mSubGraph.find(submoduleName);
    if (iter == mSubGraph.end()) {
        return nullptr;
    }
    return iter->second;
}
void Executor::setLazyComputeMode(uint32_t mode) {
    mLazyMode = mode;
}

} // namespace Express
} // namespace MNN
