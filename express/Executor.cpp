//
//  Executor.cpp
//  MNN
//
//  Created by MNN on 2019/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Executor.hpp>
#include "core/Session.hpp"
#include "core/TensorUtils.hpp"
#include "core/FileLoader.hpp"
#include "Utils.hpp"
#include <MNN/AutoTime.hpp>
#include "core/WrapExecution.hpp"
#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include <MNN/expr/ExecutorScope.hpp>
#include "core/Backend.hpp"
#include "RuntimeAttr.hpp"
#include <stack>
#define DEFAULT_BACKUP_RUNTIME_KEY (std::make_pair(MNN_FORWARD_CPU, 1))
#ifdef MNN_EXPR_ENABLE_PROFILER
#define MNN_EXPRESS_ERROR_REPORT
#endif
#define MNN_EXPRESS_OPEN_MEMORY_REUSE
namespace MNN {
namespace Express {
#ifdef MNN_EXPR_ENABLE_PROFILER
class Executor::Profiler {
public:
    void reset();
    void dump() const;
    void add(const std::string& opType, float timeInMs);
    void addFlops(const std::string& opType, float flops);
private:
    std::map<std::string, float> mTimes;
    std::map<std::string, float> mFlops;
};
void Executor::Profiler::reset() {
    mTimes.clear();
    mFlops.clear();
}
void Executor::Profiler::dump() const {
    float sumValue = 0.0f;
    for (auto iter : mTimes) {
        MNN_PRINT("%s: %f ms\n", iter.first.c_str(), iter.second);
        sumValue += iter.second;
    }
    MNN_PRINT("Total: %f ms\n", sumValue);
    sumValue = 0.0f;
    for (auto iter : mFlops) {
        MNN_PRINT("%s: %f \n", iter.first.c_str(), iter.second);
        sumValue += iter.second;
    }
    MNN_PRINT("Total flops: %f M\n", sumValue);
}
void Executor::Profiler::add(const std::string& opType, float timeInMs) {
    auto iter = mTimes.find(opType);
    if (iter == mTimes.end()) {
        mTimes[opType] = timeInMs;
        return;
    }
    iter->second += timeInMs;
}
void Executor::Profiler::addFlops(const std::string& opType, float flops) {
    auto iter = mFlops.find(opType);
    if (iter == mFlops.end()) {
        mFlops[opType] = flops;
        return;
    }
    iter->second += flops;
}
#endif

void Executor::setGlobalExecutorConfig(MNNForwardType type, const BackendConfig& config, int numberThread) {
    std::lock_guard<std::mutex> _l(mMutex);
    mFirstType = std::make_pair(type, numberThread);
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
        mFirstType = std::make_pair(type, info.numThread);

        info.user = (BackendConfig*)&config;
        std::shared_ptr<Runtime> bn(creator->onCreate(info));
        mRuntimes[mFirstType] = bn;
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
        info.mode = Backend::Info::DIRECT;
        info.numThread = numberThread;
        info.user = (BackendConfig*)&config;
        std::shared_ptr<Runtime> bn(creator->onCreate(info));
        mRuntimes[mFirstType] = bn;
    }
}

int Executor::getCurrentRuntimeStatus(RuntimeStatus statusEnum) {
    return mRuntimes[mFirstType]->onGetRuntimeStatus(statusEnum);
}

void Executor::gc(GCFlag flag) {
    int level = flag == FULL ? 100 : 0;
    for (auto& iter : mRuntimes) {
        iter.second->onGabageCollect(level);
    }
}
Executor::Executor(std::shared_ptr<Runtime> backend, MNNForwardType type, int numberThread) {
    mRuntimes.insert(std::make_pair(std::make_pair(type, numberThread), backend));
    mFirstType = std::make_pair(type, numberThread);
    if (1 != numberThread || MNN_FORWARD_CPU != type) {
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

#ifdef MNN_EXPR_ENABLE_PROFILER
    mProfiler.reset(new Profiler);
#endif
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
        req.contentNeedContent[i] = OpCommonUtils::opNeedContent(op->type(), i);
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
        gExecutor = new std::shared_ptr<Executor>(new Executor(bn, MNN_FORWARD_CPU, 1));
    });
    return *gExecutor;
}

std::shared_ptr<Executor> Executor::newExecutor(MNNForwardType type,
                                                const BackendConfig& config,
                                                int numberThread) {
    auto creator = MNNGetExtraRuntimeCreator(type);
    Backend::Info info;
    info.type = type;
    info.numThread = numberThread;
    info.user = const_cast<BackendConfig*>(&config);
    std::shared_ptr<Runtime> runtime(creator->onCreate(info));
    auto executor = new Executor(runtime, type, numberThread);
    return std::shared_ptr<Executor>(executor);
}

RuntimeInfo Executor::getRuntime() {
    RuntimeInfo info;
    auto glo = ExecutorScope::Current();
    info.second = glo->mRuntimes[DEFAULT_BACKUP_RUNTIME_KEY];
    auto cur = glo->mRuntimes[glo->mFirstType];
    info.first.insert(std::make_pair(glo->mFirstType.first, cur));
    return info;
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
    if (mode == Interpreter::Session_Input_Inside || mode == Interpreter::Session_Input_User) {
        mInside->modes.inputMode = mode;
    } else if (mode == Interpreter::Session_Output_User || mode == Interpreter::Session_Output_Inside) {
        mInside->modes.outputMode = mode;
    } else if (mode == Interpreter::Session_Backend_Auto || mode == Interpreter::Session_Backend_Fix) {
        mInside->modes.backendMode = mode;
    } else if (mode == Interpreter::Session_Debug || mode == Interpreter::Session_Release) {
        mInside->modes.callBackMode = mode;
    } else if (mode == Interpreter::Session_Resize_Direct || mode == Interpreter::Session_Resize_Defer) {
        mInside->modes.resizeMode = mode;
    }
}
void Executor::RuntimeManager::setHint(Interpreter::HintMode mode, int value) {
    mInside->modes.maxTuningNumber = value;
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
    delete mInside;
}
Executor::RuntimeManager* Executor::RuntimeManager::createRuntimeManager(const ScheduleConfig &config) {
    auto res = new RuntimeManager;
    auto glo = ExecutorScope::Current();
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
    auto iter = originRt.find(std::make_pair(compute.type, compute.numThread));
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
        originRt.insert(std::make_pair(std::make_pair(compute.type, compute.numThread), std::shared_ptr<Runtime>(newBn)));
    }
    res->mInside->mRuntime.second =  originRt[DEFAULT_BACKUP_RUNTIME_KEY];
    res->mInside->mRuntime.first.insert(std::make_pair(compute.type, originRt[std::make_pair(compute.type, compute.numThread)]));
    res->mInside->mInfo = originRt[std::make_pair(compute.type, compute.numThread)];
    res->mInside->mNumberThread = compute.numThread;
    if (nullptr != config.backendConfig) {
        res->mInside->mConfig = *config.backendConfig;
        res->mInside->mUserConfig = true;
    } else {
        res->mInside->mUserConfig = false;
    }
    return res;
}
BackendConfig* Executor::RuntimeManager::getBnConfig() {
    if (mInside->mUserConfig) {
        return &mInside->mConfig;
    }
    return nullptr;
}


void Executor::RuntimeManager::setCache(std::string cacheName) {
    mInside->mCache.reset(new Cache);
    mInside->mCache->cacheFile = cacheName;
    if (nullptr == mInside->mCache->cacheFile.c_str()) {
        MNN_ERROR("Empty cacheFile\n");
        return;
    }
    std::unique_ptr<FileLoader> loader(new FileLoader(mInside->mCache->cacheFile.c_str()));
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
    }

    mInside->mCache->lastCacheSize = mInside->mCache->cacheBuffer.size() - mInside->mCache->cacheOffset;
}

void Executor::RuntimeManager::updateCache() {
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

class Executor::ComputeCache {
public:
    void setShapeDirty();
    void setContentDirty();
    void* mapOutput(int offset, Tensor* dest);

    ~ ComputeCache();
    ComputeCache(std::shared_ptr<Backend> backend, std::shared_ptr<Backend> backupBackend);

    ErrorCode compute();
    ErrorCode resize();
    ErrorCode resizeImpl();
    std::pair<std::shared_ptr<Backend>, std::shared_ptr<Backend>> bakcends() const {
        return std::make_pair(mBackend, mBackupBackend);
    }
private:
    std::set<std::shared_ptr<ComputeCache>> mInputs;
    std::vector<Tensor*> mOutputs;
    std::vector<std::shared_ptr<Unit>> mUnits;
    std::shared_ptr<Backend> mBackend;
    std::shared_ptr<Backend> mBackupBackend;
    std::set<std::shared_ptr<Expr::Inside>> mInputInside;
    friend class Executor;
    bool mContentDirty = true;
    bool mShapeDirty = true;
    GeometryComputer::Context mContext;
    std::vector<CommandBuffer> mCmdBuffer;
    std::map<const Op*, std::shared_ptr<Execution>> mCacheExes;
    Runtime::CompilerType mCompilerType;
    std::map<Tensor*, std::shared_ptr<Tensor>> mCacheConstTensors;
#ifdef MNN_EXPRESS_MEMLEAK_DEBUG
    static int gInstanceCount;
#endif
};
#ifdef MNN_EXPRESS_MEMLEAK_DEBUG
int Executor::ComputeCache::gInstanceCount = 0;
#endif
void Executor::setShapeDirty(ComputeCache* cache) {
    cache->setShapeDirty();
}
void Executor::setContentDirty(ComputeCache* cache) {
    cache->setContentDirty();
}
void* Executor::mapOutput(ComputeCache* cache, int offset, Tensor* dest) {
    return cache->mapOutput(offset, dest);
}

struct Executor::Unit {
    std::vector<Tensor*> inputs;
    std::vector<Tensor*> outputs;
    const Op* op;
    std::shared_ptr<BufferStorage> opStorage;
    std::weak_ptr<Expr::Inside> inside;
    std::vector<std::shared_ptr<Tensor>> outputContents;
};
Tensor* Executor::getOutput(ComputeCache* cache, int offset) {
    return cache->mOutputs[offset];
}
std::pair<std::shared_ptr<Backend>, std::shared_ptr<Backend>> Executor::getBackends(ComputeCache* cache) {
    return cache->bakcends();
}

void* Executor::ComputeCache::mapOutput(int offset, Tensor* dest) {
    auto tensor = mOutputs[offset];
    if (0 == tensor->deviceId()) {
        auto ptr =  tensor->host<void>();
        Utils::releaseMemoryForHostTensor(dest);
        TensorUtils::getDescribe(dest)->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
        dest->buffer().host = (uint8_t*)ptr;
        //MNN_ASSERT(nullptr != ptr);
        return ptr;
    }
    Utils::allocMemoryForHostTensor(dest);
    tensor->copyToHostTensor(dest);
    MNN_ASSERT(nullptr != dest->host<void>());
    return dest->host<void>();
}

void Executor::ComputeCache::setShapeDirty() {
    mShapeDirty = true;
}

void Executor::ComputeCache::setContentDirty() {
    mContentDirty = true;
}

Executor::ComputeCache::ComputeCache(std::shared_ptr<Backend> backend, std::shared_ptr<Backend> backupBackend) : mContext(backupBackend, true, backend->type()) {
    mBackend = backend;
    mBackupBackend = backupBackend;
#ifdef MNN_EXPRESS_MEMLEAK_DEBUG
    gInstanceCount++;
    FUNC_PRINT(gInstanceCount);
#endif
}
Executor::ComputeCache::~ComputeCache() {
    mUnits.clear();
    mCacheExes.clear();
#ifdef MNN_EXPRESS_MEMLEAK_DEBUG
    gInstanceCount--;
    FUNC_PRINT(gInstanceCount);
#endif
}
ErrorCode Executor::ComputeCache::compute() {
    std::stack<ComputeCache*> dfsStack;
    std::set<ComputeCache*> visited;
    dfsStack.push(this);
    while (!dfsStack.empty()) {
        //printf("stcak = %d\n", dfsStack.size());
        auto cache = dfsStack.top();
        if (cache->mShapeDirty) {
            auto code = cache->resize();
            if (NO_ERROR != code) {
                cache->mShapeDirty = true;
                return code;
            }
        }
        if (!cache->mContentDirty) {
            visited.insert(cache);
            dfsStack.pop();
            continue;
        }
        for (auto& c : cache->mInputInside) {
            if (c->mContentDirty) {
                return CALL_BACK_STOP;
            }
        }
        auto hasUnvisitInput = [&] () {
            for (auto c : cache->mInputs) {
                if (visited.find(c.get()) == visited.end()) {
                    return true;
                }
            }
            return false;
        };
        if (hasUnvisitInput()) {
            for (auto c : cache->mInputs) {
                dfsStack.push(c.get());
            }
        } else {
            visited.insert(cache);
            dfsStack.pop();
            cache->mBackend->onExecuteBegin();
            cache->mBackupBackend->onExecuteBegin();
            for (auto& buffer : cache->mCmdBuffer) {
                for (int i=0; i<buffer.command.size(); ++i) {
#ifdef MNN_EXPR_ENABLE_PROFILER
                    Timer autoTime;
#endif
                    auto& iter = *buffer.command[i];
                    auto code = iter.execution->onExecute(iter.inputs, iter.outputs);
                    if (NO_ERROR != code) {
#ifdef MNN_EXPRESS_ERROR_REPORT
                        auto op = iter.op;
                        MNN_ERROR("Error to compute for %s, \n", EnumNameOpType(op->type()));
#endif
                        cache->mBackend->onExecuteEnd();
                        return code;
                    }
#ifdef MNN_EXPR_ENABLE_PROFILER
                    float costTime = (float)autoTime.durationInUs() / (float)1000;
                    auto op = iter.op;
                    ExecutorScope::Current()->addOpCostTime((int)op->type(), costTime);
#endif
                }
            }
            cache->mBackend->onExecuteEnd();
            cache->mBackupBackend->onExecuteEnd();
            cache->mContentDirty = false;
        }
    }
    return NO_ERROR;
}
ErrorCode Executor::ComputeCache::resizeImpl() {
    mShapeDirty = false;
    mCmdBuffer.resize(mUnits.size());
    /** Encoder Begin */
    {
#ifdef MNN_EXPR_ENABLE_PROFILER
        {
        Timer autoTime;
#endif
        mContext.clear();
#ifdef MNN_EXPR_ENABLE_PROFILER
        float costTime = (float)autoTime.durationInUs() / (float)1000;
        ExecutorScope::Current()->addOpCostTime((int)OpType_While, costTime);
        }
#endif
        CommandBuffer buffer;
        for (int unitIndex = 0; unitIndex < mUnits.size(); ++unitIndex) {
            auto& iter = *mUnits[unitIndex];
            auto inside = iter.inside.lock();
            if (nullptr == inside || inside->mInfoDirty) {
                mShapeDirty = true;
                continue;
            }
            buffer.command.clear();
            buffer.extras.clear();
            // Check zero shape
            bool zeroShape = false;
            for (int i=0; i<iter.outputs.size(); ++i) {
                TensorUtils::copyShape(inside->mOutputTensors[i], iter.outputs[i], true);
                TensorUtils::getDescribe(iter.outputs[i])->tensorArrayAttr = TensorUtils::getDescribe(inside->mOutputTensors[i])->tensorArrayAttr;
                auto t = iter.outputs[i];
                iter.outputs[i]->buffer().type = inside->mOutputTensors[i]->buffer().type;
                auto des = TensorUtils::getDescribe(iter.outputs[i]);
                if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND) {
                    des->backend = nullptr;
                    des->mem.reset(nullptr);
                }
                des->regions.clear();
                for (int v=0; v<t->dimensions(); ++v) {
                    if (t->length(v) == 0) {
                        zeroShape = true;
                        break;
                    }
                    if (t->length(v) < 0) {
                        return INPUT_DATA_ERROR;
                    }
                }
            }
            if (zeroShape) {
                // FIXME: for multi output and one tensor zero shape should support
                continue;
            }
#ifdef MNN_EXPR_ENABLE_PROFILER
            {
            Timer autoTime;
#endif
            auto geo = GeometryComputer::search(iter.op->type(), mCompilerType);
            geo->onCompute(iter.op, iter.inputs, iter.outputs, mContext, buffer);
            mCmdBuffer[unitIndex].command.clear();
            mCmdBuffer[unitIndex].extras.clear();
            GeometryComputerUtils::makeRaster(buffer, mCmdBuffer[unitIndex], mContext);
            for (auto out : iter.outputs) {
                if (TensorUtils::getDescribe(out)->usage == Tensor::InsideDescribe::OUTPUT) {
                    mContext.getRasterCacheCreateRecursive(out, mCmdBuffer[unitIndex]);
                }
            }
#ifdef MNN_EXPR_ENABLE_PROFILER
            float costTime = (float)autoTime.durationInUs() / (float)1000;
                ExecutorScope::Current()->addOpCostTime((int)iter.op->type(), costTime);
            }
#endif
        }
#ifdef MNN_EXPR_ENABLE_PROFILER
        {
        Timer autoTime;
#endif
        for (int unitIndex = 0; unitIndex < mUnits.size(); ++unitIndex) {
            auto& iter = *mUnits[unitIndex];
            auto inside = iter.inside.lock();
            if (nullptr == inside || inside->mInfoDirty) {
                mShapeDirty = true;
                continue;
            }
        }

#ifdef MNN_EXPR_ENABLE_PROFILER
        float costTime = (float)autoTime.durationInUs() / (float)1000;
        ExecutorScope::Current()->addOpCostTime((int)OpType_If, costTime);
        }
#endif
    }
    for (auto& buffer : mCmdBuffer) {
        for (int k=0; k<buffer.command.size(); ++k) {
            auto& cmd = *buffer.command[k];
            auto op = cmd.op;
            for (auto v = 0; v<cmd.inputs.size(); ++v) {
                if (!OpCommonUtils::opNeedContent(op->type(), v)) {
                    continue;
                }
                auto des = TensorUtils::getDescribe(cmd.inputs[v]);
                if (op->type() == OpType_Raster) {
                    for (auto& s : des->regions) {
                        auto subDes = TensorUtils::getDescribe(s.origin);
                        if (subDes->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && subDes->usage == Tensor::InsideDescribe::NORMAL) {
                            subDes->useCount+=1;
                        }
                    }
                } else {
                    if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && des->usage == Tensor::InsideDescribe::NORMAL) {
                        des->useCount+=1;
                    }
                }
            }
        }
    }
    /** Encoder End */

    /** Prepare Begin */
    mBackend->onClearBuffer();
    mBackupBackend->onClearBuffer();
    mBackend->onResizeBegin();
    for (auto& buffer : mCmdBuffer) {
        for (int k=0; k<buffer.command.size(); ++k) {
            auto& cmd = *buffer.command[k];
            auto op = cmd.op;
            bool origin = true;
    #ifdef MNN_EXPR_ENABLE_PROFILER
            Timer autoTime;
    #endif
            cmd.executionOrigin = nullptr;
            bool cacheed = false;
            if (!mCacheExes.empty() && origin) {
                auto iter = mCacheExes.find(op);
                if (iter != mCacheExes.end()) {
                    cmd.executionOrigin = iter->second;
                    cacheed = true;
                }
            }
            if (nullptr == cmd.executionOrigin) {
                cmd.executionOrigin.reset(mBackend->onCreate(cmd.inputs, cmd.outputs, op));
                if (nullptr == cmd.executionOrigin) {
                    cmd.executionOrigin.reset(mBackupBackend->onCreate(cmd.inputs, cmd.outputs, op));
                }
                if (nullptr == cmd.executionOrigin) {
                    return NOT_SUPPORT;
                }
            }
            // Check if need wrap
            bool wrap = false;
            auto bn = cmd.executionOrigin->backend();
            auto iterType = bn->type();
            for (int i=0; i<cmd.inputs.size(); ++i) {
                if (!OpCommonUtils::opNeedContent(op->type(), i)) {
                    continue;
                }
                auto inpDes = TensorUtils::getDescribe(cmd.inputs[i]);
                if (op->type() == OpType_Raster) {
                    for (auto& r : inpDes->regions) {
                        MNNForwardType type = MNN_FORWARD_CPU;
                        auto origin     = r.origin;
                        if (WrapExecution::needWrap(origin, bn)) {
                            auto newTensor = WrapExecution::copyConstCache(origin, bn, mCacheConstTensors);
                            if (nullptr != newTensor) {
                                r.origin = newTensor;
                            } else {
                                wrap = true;
                            }
                        }
                    }
                } else {
                    auto t = cmd.inputs[i];
                    if (WrapExecution::needWrap(t, bn)) {
                        auto newTensor = WrapExecution::copyConstCache(t, bn, mCacheConstTensors);
                        if (nullptr != newTensor) {
                            cmd.inputs[i] = newTensor;
                        } else {
                            wrap = true;
                        }
                    }
                }
            }
            if (wrap && (!cacheed)) {
                cmd.execution.reset(new WrapExecution(mBackupBackend.get(), cmd.executionOrigin, false));
            } else {
                cmd.execution = cmd.executionOrigin;
            }
            if ((op->type() == OpType_Convolution && cmd.inputs.size() == 1)) {
                // TODO: Support Other op's cache
                mCacheExes.insert(std::make_pair(op, cmd.executionOrigin));
            }
            for (auto t : cmd.outputs) {
                auto des = TensorUtils::getDescribe(t);
                if (nullptr == des->backend) {
                    TensorUtils::setLinearLayout(t);
                    auto allocType = Backend::DYNAMIC;
                    if (des->usage == Tensor::InsideDescribe::OUTPUT) {
                        allocType = Backend::STATIC;
                    }
                    auto res = bn->onAcquireBuffer(t, allocType);
                    des->backend = bn;
                    if (!res) {
                        return OUT_OF_MEMORY;
                    }
                }
            }

            auto code= cmd.execution->onResize(cmd.inputs, cmd.outputs);
            if (NO_ERROR != code) {
                return code;
            }
            bool isRaster = cmd.inputs.size() == 1 && cmd.inputs[0] == cmd.outputs[0];
            for (auto v = 0; v<cmd.inputs.size(); ++v) {
                if (!OpCommonUtils::opNeedContent(op->type(), v)) {
                    continue;
                }
                auto t = cmd.inputs[v];
                auto des = TensorUtils::getDescribe(t);
                if (!isRaster) {
                    if (des->usage == Tensor::InsideDescribe::NORMAL) {
                        des->useCount-=1;
                        if (0 == des->useCount && nullptr != des->backend) {
                            des->backend->onReleaseBuffer(t, Backend::DYNAMIC);
                        }
                    }
                } else {
                    for (auto& s : des->regions) {
                        auto subDes = TensorUtils::getDescribe(s.origin);
                        if (subDes->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && subDes->usage == Tensor::InsideDescribe::NORMAL) {
                            subDes->useCount-=1;
                            if (0 == subDes->useCount && nullptr != subDes->backend) {
                                subDes->backend->onReleaseBuffer(s.origin, Backend::DYNAMIC);
                            }
                        }
                    }
                }
            }
    #ifdef MNN_EXPR_ENABLE_PROFILER
            float costTime = (float)autoTime.durationInUs() / (float)1000;
            ExecutorScope::Current()->addOpCostTime((int)op->type(), costTime);
    #endif
        }
    }
    for (auto& buffer : mCmdBuffer) {
        for (int k=0; k<buffer.command.size(); ++k) {
            auto& cmd = *buffer.command[k];
            for (auto t : cmd.outputs) {
                if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::OUTPUT) {
                    continue;
                }
                TensorUtils::getDescribe(t)->mem.reset(nullptr);
            }
        }
    }
    mBackend->onResizeEnd();

    /** Prepare End */

    mContentDirty = true;
    return NO_ERROR;
}
ErrorCode Executor::ComputeCache::resize() {
    std::stack<ComputeCache*> dfsStack;
    std::set<ComputeCache*> visited;
    dfsStack.push(this);
    while (!dfsStack.empty()) {
        auto cache = dfsStack.top();
        if (!cache->mShapeDirty) {
            visited.insert(cache);
            dfsStack.pop();
            continue;
        }
        for (auto& c : cache->mInputInside) {
            if (c->mInfoDirty) {
                return CALL_BACK_STOP;
            }
        }
        auto hasUnvisitInput = [&] () {
            for (auto c : cache->mInputs) {
                if (visited.find(c.get()) == visited.end()) {
                    return true;
                }
            }
            return false;
        };
        if (hasUnvisitInput()) {
            for (auto c : cache->mInputs) {
                dfsStack.push(c.get());
            }
        } else {
            visited.insert(cache);
            dfsStack.pop();
            auto code = cache->resizeImpl();
            if (code != NO_ERROR) {
                return code;
            }
        }
    }
    return NO_ERROR;
}

static void _collectExecuteUnit(std::vector<std::shared_ptr<Executor::Unit>>& dest, EXPRP expr) {
    std::stack<EXPRP> dfsStack;
    std::set<EXPRP> visited;
    dfsStack.push(expr);
    while (!dfsStack.empty()) {
        auto expr = dfsStack.top();
        auto& inputs = expr->inputs();
        auto& req = expr->inside()->mReq.contentNeedContent;
        MNN_ASSERT(inputs.size() == req.size());
        auto hasUnvisitInput = [&]() {
            for (int i = 0; i < inputs.size(); ++i) {
                if (!req[i]) {
                    continue;
                }
                auto inputExpr = inputs[i]->expr();
                auto unit = inputExpr.first->inside()->mUnit;
                if (nullptr == unit) {
                    continue;
                }
                auto inputCache = inputExpr.first->inside()->mCache;
                if (nullptr != inputCache) {
                    continue;
                }
                if (visited.find(inputExpr.first) != visited.end()) {
                    continue;
                }
                return true;
            }
            return false;
        };
        // if no input or input has visit, do visit
        if (!hasUnvisitInput()) {
            auto visitFunc = [&]() {
                visited.insert(expr);
                auto unit = expr->inside()->mUnit;
                if (nullptr == unit) {
                    return;
                }
                dest.emplace_back(std::move(unit));
                expr->inside()->mUnit = nullptr;
            };
            visitFunc();
            dfsStack.pop();
        } else {
            for (int i = 0; i < inputs.size(); ++i) {
                if (!req[i]) {
                    continue;
                }
                auto inputExpr = inputs[i]->expr();
                auto unit = inputExpr.first->inside()->mUnit;
                if (nullptr == unit) {
                    continue;
                }
                auto inputCache = inputExpr.first->inside()->mCache;
                if (nullptr != inputCache) {
                    continue;
                }
                dfsStack.push(inputExpr.first);
            }
        }
    }
}

void Executor::_create(const std::vector<EXPRP>& outputs, std::set<std::shared_ptr<Executor::ComputeCache>>&& inputCaches, std::set<std::shared_ptr<Expr::Inside>>&& inputNode, bool forceCPU) {
    std::vector<EXPRP> packed;
    for (auto expr : outputs) {
        auto cache = expr->inside()->mCache;
        if (nullptr != cache) {
            continue;
        }
        if (nullptr != expr->get()) {
            packed.emplace_back(expr);
            continue;
        }
    }
    if (packed.empty()) {
        return;
    }
    //MNN_PRINT("Create %p begin\n", packed[0].get());
    std::shared_ptr<Backend> cacheBn;
    std::shared_ptr<Backend> cacheBackupBn;
    BackendConfig defaultConfig;
    defaultConfig.flags = 4;
    auto backupRuntime = mRuntimes[DEFAULT_BACKUP_RUNTIME_KEY];
    auto mainRuntime = mRuntimes[mFirstType];
    if (forceCPU) {
        cacheBn.reset(backupRuntime->onCreate(&defaultConfig));
        cacheBackupBn = cacheBn;
    } else {
        cacheBn.reset(mainRuntime->onCreate());
        cacheBackupBn.reset(backupRuntime->onCreate(&defaultConfig));
    }
    std::shared_ptr<ComputeCache> packedCache(new ComputeCache(cacheBn, cacheBackupBn));
    packedCache->mCompilerType = mainRuntime->onGetCompilerType();
    packedCache->mInputs = std::move(inputCaches);
    packedCache->mInputInside = std::move(inputNode);
    for (auto expr : packed) {
        expr->inside()->mCacheOffset = (int)packedCache->mOutputs.size();
        MNN_ASSERT(expr->inside()->mUnit != nullptr);
        auto& originOutputs = expr->inside()->mUnit->outputs;
        for (auto t : originOutputs) {
            packedCache->mOutputs.emplace_back(t);
            TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
        }
    }
    for (auto expr : packed) {
        _collectExecuteUnit(packedCache->mUnits, expr);
    }
    for (auto expr : packed) {
        expr->inside()->mCache = packedCache;
    }
    //MNN_PRINT("Create %p End\n", packed[0].get());
}

void Executor::_visit(EXPRP expr, std::set<std::shared_ptr<Executor::ComputeCache>>& inputCaches, std::set<std::shared_ptr<Expr::Inside>>& inputNode) {
    std::stack<EXPRP> dfsStack;
    std::set<EXPRP> visited;
    dfsStack.push(expr);
    while (!dfsStack.empty()) {
        auto expr = dfsStack.top();
        auto& inputs = expr->inputs();
        auto& req = expr->inside()->mReq.contentNeedContent;
        MNN_ASSERT(inputs.size() == req.size());
        auto hasUnvisitInput = [&]() {
            for (int i = 0; i < inputs.size(); ++i) {
                if (!req[i]) {
                    continue;
                }
                auto inputExpr = inputs[i]->expr();
                if (nullptr != inputExpr.first->inside()->mUnit) {
                    continue;
                }
                auto inputCache = inputExpr.first->inside()->mCache;
                if (nullptr != inputCache) {
                    continue;
                }
                if (visited.find(inputExpr.first) != visited.end()) {
                    continue;
                }
                return true;
            }
            return false;
        };
        // if no input or input has visit, do visit
        if (!hasUnvisitInput()) {
            auto visitFunc = [&]() {
                visited.insert(expr);
                auto op = expr->get();
                if (nullptr == op) {
                    return;
                }
                if (nullptr != expr->inside()->mUnit) {
                    return;
                }
                std::shared_ptr<Unit> unitP(new Unit);
                Unit& unit = *unitP;
                unit.op = expr->get();
                unit.opStorage = expr->extra();
                unit.inside = std::weak_ptr<Expr::Inside>(expr->inside());
                unit.inputs.resize(inputs.size());
                unit.outputs.resize(expr->inside()->mOutputTensors.size());
                unit.outputContents.resize(unit.outputs.size());
                for (int i=0; i<unit.outputs.size(); ++i) {
                    unit.outputContents[i].reset(new Tensor);
                    unit.outputs[i] = unit.outputContents[i].get();
                }
                for (int i=0; i<inputs.size(); ++i) {
                    auto inputExpr = inputs[i]->expr();
                    unit.inputs[i] = inputExpr.first->inside()->mOutputTensors[inputExpr.second];
                    if (!req[i]) {
                        // The compute don't need it
                        continue;
                    }
                    if (inputExpr.first->get() == nullptr) {
                        if (inputExpr.first->inputType() == VARP::INPUT) {
                            inputNode.insert(inputExpr.first->inside());
                        }
                        continue;
                    }
                    auto inputUnit = inputExpr.first->inside()->mUnit;
                    if (nullptr != inputUnit) {
                        unit.inputs[i] = inputUnit->outputs[inputExpr.second];
                        continue;
                    }
                    MNN_ASSERT(nullptr != inputExpr.first->inside()->mCache);
                    inputCaches.insert(inputExpr.first->inside()->mCache);
                    auto offset = inputExpr.second + inputExpr.first->inside()->mCacheOffset;
                    unit.inputs[i] = inputExpr.first->inside()->mCache->mOutputs[offset];
                }
                MNN_ASSERT(expr->inside()->mUnit == nullptr);
                //MNN_PRINT("Create %p, %s\n", expr.get(), EnumNameOpType(expr->get()->type()));
                expr->inside()->mUnit = unitP;
            };
            visitFunc();
            dfsStack.pop();
        } else {
            for (int i = 0; i < inputs.size(); ++i) {
                if (!req[i]) {
                    continue;
                }
                auto inputExpr = inputs[i]->expr();
                if (nullptr != inputExpr.first->inside()->mUnit) {
                    continue;
                }
                auto inputCache = inputExpr.first->inside()->mCache;
                if (nullptr != inputCache) {
                    inputCaches.insert(inputCache);
                    continue;
                }
                dfsStack.push(inputExpr.first);
            }
        }
    }
}
void Executor::_makeCache(const std::vector<EXPRP>& expr, bool forceCPU) {
    std::set<std::shared_ptr<Executor::ComputeCache>> inputCaches;
    std::set<std::shared_ptr<Expr::Inside>> inputNode;
    for (auto e : expr) {
        _visit(e, inputCaches, inputNode);
    }
    _create(expr, std::move(inputCaches), std::move(inputNode), forceCPU);
}

void Executor::makeCache(const std::vector<EXPRP>& expr, bool forceCPU) {
    std::lock_guard<std::mutex> _l(mMutex);
    //FUNC_PRINT(mCaches.size());
    _makeCache(expr, forceCPU);
}
void Executor::addOpCostTime(int op, float costTime) {
#ifdef MNN_EXPR_ENABLE_PROFILER
    auto opType = MNN::EnumNameOpType((OpType)op);
    if (nullptr == opType) {
        return;
    }
    mProfiler->add(opType, costTime);
#endif
}
void Executor::addOpCostTime(const std::string& type, float costTime) {
#ifdef MNN_EXPR_ENABLE_PROFILER
    mProfiler->add(type, costTime);
#endif
}
void Executor::addOpFlops(const std::string& type, float flops) {
#ifdef MNN_EXPR_ENABLE_PROFILER
    mProfiler->addFlops(type, flops);
#endif
}


ErrorCode Executor::runCache(std::shared_ptr<ComputeCache> cache) {
    std::lock_guard<std::mutex> _l(mMutex);
    return cache->compute();
}
void Executor::resetProfile() {
#ifdef MNN_EXPR_ENABLE_PROFILER
    mProfiler->reset();
#endif
}
void Executor::dumpProfile() {
#ifdef MNN_EXPR_ENABLE_PROFILER
    mProfiler->dump();
#endif
}

} // namespace Express
} // namespace MNN
