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
#include "Utils.hpp"
#include <MNN/AutoTime.hpp>
#include "core/WrapExecution.hpp"
#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include <MNN/expr/ExecutorScope.hpp>
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
    mRuntime.first = bn;
    mRuntime.second = type;
}

void Executor::gc(GCFlag flag) {
    if (FULL == flag) {
        mBackupRuntime.first->onGabageCollect(100);
        mRuntime.first->onGabageCollect(100);
    } else {
        mBackupRuntime.first->onGabageCollect(0);
        mRuntime.first->onGabageCollect(0);
    }
}
Executor::Executor(std::shared_ptr<Runtime> backend, MNNForwardType type) {
    mRuntime.first = backend;
    mRuntime.second = type;
    Backend::Info info;
    info.type = MNN_FORWARD_CPU;
    auto cre = MNNGetExtraRuntimeCreator(MNN_FORWARD_CPU);
    info.mode = Backend::Info::DIRECT;
    info.numThread = 1;
    mBackupRuntime.first.reset(cre->onCreate(info));
    mBackupRuntime.second = MNN_FORWARD_CPU;

#ifdef MNN_EXPR_ENABLE_PROFILER
    mProfiler.reset(new Profiler);
#endif
}
Executor::~Executor(){
    mRuntime.first = nullptr;
    mBackupRuntime.first = nullptr;
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
std::shared_ptr<Executor> Executor::getGlobalExecutor() {
    static std::shared_ptr<Executor> gExecutor;
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
        gExecutor.reset(new Executor(bn, MNN_FORWARD_CPU));
    });
    return gExecutor;
}

std::shared_ptr<Executor> Executor::newExecutor(MNNForwardType type,
                                                const BackendConfig& config,
                                                int numberThread) {
    auto creator = MNNGetExtraRuntimeCreator(type);
    Backend::Info info;
    info.type = type;
    info.numThread = numberThread;
    info.user = const_cast<BackendConfig*>(&config);
    std::shared_ptr<Runtime> bn(creator->onCreate(info));
    return std::shared_ptr<Executor>(new Executor(bn, type));
}

RuntimeInfo Executor::getRuntime() {
    RuntimeInfo info;
    auto glo = ExecutorScope::Current();
    info.second = glo->mBackupRuntime.first;
    info.first.insert(std::make_pair(glo->mRuntime.second, glo->mRuntime.first));
    return info;
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
    CommandBuffer mCmdBuffer;
    std::vector<std::shared_ptr<Execution>> mExecutions;
    std::map<const Op*, std::shared_ptr<Execution>> mCacheExes;
    Runtime::CompilerType mCompilerType;
};
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
    std::weak_ptr<Expr::Inside> inside;
    std::vector<std::shared_ptr<Tensor>> outputContents;
};
Tensor* Executor::getOutput(ComputeCache* cache, int offset) {
    return cache->mOutputs[offset];
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
}
Executor::ComputeCache::~ComputeCache() {
    mUnits.clear();
    mCacheExes.clear();
}
ErrorCode Executor::ComputeCache::compute() {
    if (mShapeDirty) {
        auto code = resize();
        if (NO_ERROR != code) {
            return code;
        }
    }
    if (!mContentDirty) {
        return NO_ERROR;
    }
    for (auto& c : mInputInside) {
        if (c->mContentDirty) {
            return CALL_BACK_STOP;
        }
    }
    for (auto c : mInputs) {
        auto code = c->compute();
        if (NO_ERROR != code) {
            return code;
        }
    }
    mBackend->onExecuteBegin();
    mBackupBackend->onExecuteBegin();
    MNN_ASSERT(mExecutions.size() == mCmdBuffer.command.size());
    for (int i=0; i<mCmdBuffer.command.size(); ++i) {
#ifdef MNN_EXPR_ENABLE_PROFILER
        Timer autoTime;
#endif
        auto& iter = mCmdBuffer.command[i];
        auto code = mExecutions[i]->onExecute(iter.inputs, iter.outputs);
        if (NO_ERROR != code) {
#ifdef MNN_EXPRESS_ERROR_REPORT
            auto op = iter.buffer.empty() ? iter.op : flatbuffers::GetRoot<Op>(iter.buffer.data());
            MNN_ERROR("Error to compute for %s, \n", EnumNameOpType(op->type()));
#endif
            mBackend->onExecuteEnd();
            return code;
        }
#ifdef MNN_EXPR_ENABLE_PROFILER
        float costTime = (float)autoTime.durationInUs() / (float)1000;
        auto op = iter.op;
        if (!iter.buffer.empty()) {
            op = flatbuffers::GetMutableRoot<Op>(iter.buffer.data());
        }
        ExecutorScope::Current()->addOpCostTime((int)op->type(), costTime);
#endif
    }
    mBackend->onExecuteEnd();
    mBackupBackend->onExecuteEnd();
    mContentDirty = false;
    return NO_ERROR;
}
ErrorCode Executor::ComputeCache::resize() {
    if (!mShapeDirty) {
        return NO_ERROR;
    }
    for (auto& c : mInputInside) {
        if (c->mInfoDirty) {
            return CALL_BACK_STOP;
        }
    }
    for (auto c : mInputs) {
        auto code = c->resize();
        if (NO_ERROR != code) {
            return code;
        }
    }
    mShapeDirty = false;
    /** Encoder Begin */
    {
#ifdef MNN_EXPR_ENABLE_PROFILER
        {
        Timer autoTime;
#endif
        mCmdBuffer.command.clear();
        mCmdBuffer.extras.clear();
        mBackend->onClearBuffer();
        mBackupBackend->onClearBuffer();
        mExecutions.clear();
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
            // Check zero shape
            bool zeroShape = false;
            for (int i=0; i<iter.outputs.size(); ++i) {
                TensorUtils::copyShape(inside->mOutputTensors[i], iter.outputs[i], true);
                auto t = iter.outputs[i];
                // FIXME: Find better way to may compability for old model
                /**
                 For Convolution of 2D / 3D Tensor(Dense / 1D Convolution)
                 Because of old code, we will acces dim[2] / dim[3] to get width and height
                 Set the lenght to 1 for compability
                 */
                for (int v=t->dimensions(); v<4; ++v) {
                    t->setLength(v, 1);
                }
                iter.outputs[i]->buffer().type = inside->mOutputTensors[i]->buffer().type;
                auto des = TensorUtils::getDescribe(iter.outputs[i]);
                if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND) {
                    des->backend = nullptr;
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
            geo->compute(iter.op, iter.inputs, iter.outputs, mContext, buffer);
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
        GeometryComputerUtils::makeRaster(buffer, mCmdBuffer, mContext);
#ifdef MNN_EXPR_ENABLE_PROFILER
        float costTime = (float)autoTime.durationInUs() / (float)1000;
        ExecutorScope::Current()->addOpCostTime((int)OpType_If, costTime);
        }
#endif
    }
    for (int k=0; k<mCmdBuffer.command.size(); ++k) {
        auto& cmd = mCmdBuffer.command[k];
        auto op = cmd.op;
        if (!cmd.buffer.empty()) {
            op = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
        }
        for (auto v = 0; v<cmd.inputs.size(); ++v) {
            if (!OpCommonUtils::opNeedContent(op->type(), v)) {
                continue;
            }
            auto des = TensorUtils::getDescribe(cmd.inputs[v]);
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && des->usage == Tensor::InsideDescribe::NORMAL) {
                des->useCount+=1;
                continue;;
            }
            for (auto& s : des->regions) {
                auto subDes = TensorUtils::getDescribe(s.origin);
                if (subDes->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && subDes->usage == Tensor::InsideDescribe::NORMAL) {
                    subDes->useCount+=1;
                }
            }
        }
    }
    /** Encoder End */

    /** Prepare Begin */
    mBackend->onResizeBegin();
    mExecutions.resize(mCmdBuffer.command.size());
    for (int k=0; k<mCmdBuffer.command.size(); ++k) {
        auto& cmd = mCmdBuffer.command[k];
        auto op = cmd.op;
        bool origin = true;
        if (!cmd.buffer.empty()) {
            origin = false;
            op = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
        }
#ifdef MNN_EXPR_ENABLE_PROFILER
        Timer autoTime;
#endif
        mExecutions[k] = nullptr;
        bool cacheed = false;
        if (!mCacheExes.empty() && origin) {
            auto iter = mCacheExes.find(op);
            if (iter != mCacheExes.end()) {
                mExecutions[k] = iter->second;
                cacheed = true;
            }
        }
        if (nullptr == mExecutions[k]) {
            mExecutions[k].reset(mBackend->onCreate(cmd.inputs, cmd.outputs, op));
            if (nullptr == mExecutions[k]) {
                mExecutions[k].reset(mBackupBackend->onCreate(cmd.inputs, cmd.outputs, op));
            }
            if (nullptr == mExecutions[k]) {
                return NOT_SUPPORT;
            }
        }
        // Check if need wrap
        bool needWrap = false;
        auto bn = mExecutions[k]->backend();
        auto iterType = bn->type();
        for (int i=0; i<cmd.inputs.size(); ++i) {
            if (!OpCommonUtils::opNeedContent(op->type(), i)) {
                continue;
            }
            auto inpDes = TensorUtils::getDescribe(cmd.inputs[i]);
            if (inpDes->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                for (auto& reg : inpDes->regions) {
                    needWrap = needWrap || WrapExecution::needWrap(reg.origin, bn);
                }
            } else {
                needWrap = needWrap || WrapExecution::needWrap(cmd.inputs[i], bn);
            }
            if (needWrap) {
                break;
            }
        }
        if (needWrap && (!cacheed)) {
            mExecutions[k].reset(new WrapExecution(mBackupBackend.get(), mExecutions[k], false));
        }
        if ((op->type() == OpType_Convolution && cmd.inputs.size() == 1)) {
            // TODO: Support Other op's cache
            mCacheExes.insert(std::make_pair(op, mExecutions[k]));
        }
        for (auto t : cmd.outputs) {
            auto des = TensorUtils::getDescribe(t);
            if (nullptr == des->backend) {
                TensorUtils::setLinearLayout(t);
                auto res = bn->onAcquireBuffer(t, Backend::DYNAMIC);
                des->backend = bn;
                if (!res) {
                    return OUT_OF_MEMORY;
                }
            }
        }
        auto code= mExecutions[k]->onResize(cmd.inputs, cmd.outputs);
        if (NO_ERROR != code) {
            return code;
        }
        for (auto v = 0; v<cmd.inputs.size(); ++v) {
            if (!OpCommonUtils::opNeedContent(op->type(), v)) {
                continue;
            }
            auto t = cmd.inputs[v];
            auto des = TensorUtils::getDescribe(t);
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND) {
                if (des->usage == Tensor::InsideDescribe::NORMAL) {
                    des->useCount-=1;
                    if (0 == des->useCount && nullptr != des->backend) {
                        des->backend->onReleaseBuffer(t, Backend::DYNAMIC);
                    }
                }
            }
            for (auto& s : des->regions) {
                auto subDes = TensorUtils::getDescribe(s.origin);
                MNN_ASSERT(subDes->regions.empty());
                if (subDes->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND && subDes->usage == Tensor::InsideDescribe::NORMAL) {
                    subDes->useCount-=1;
                    if (0 == subDes->useCount && nullptr != subDes->backend) {
                        subDes->backend->onReleaseBuffer(s.origin, Backend::DYNAMIC);
                    }
                }
            }
        }
#ifdef MNN_EXPR_ENABLE_PROFILER
        float costTime = (float)autoTime.durationInUs() / (float)1000;
        ExecutorScope::Current()->addOpCostTime((int)op->type(), costTime);
#endif
    }
    mBackend->onResizeEnd();

    /** Prepare End */

    mContentDirty = true;
    return NO_ERROR;
}

static void _collectExecuteUnit(std::vector<std::shared_ptr<Executor::Unit>>& dest, EXPRP expr) {
    auto& inputs = expr->inputs();
    auto& req = expr->inside()->mReq.contentNeedContent;
    MNN_ASSERT(inputs.size() == req.size());
    
    for (int i=0; i<inputs.size(); ++i) {
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
        _collectExecuteUnit(dest, inputExpr.first);
    }
    auto unit = expr->inside()->mUnit;
    if (nullptr == unit) {
        return;
    }
    dest.emplace_back(std::move(unit));
    expr->inside()->mUnit = nullptr;
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
    if (forceCPU) {
        cacheBn.reset(mBackupRuntime.first->onCreate(&defaultConfig));
        cacheBackupBn = cacheBn;
    } else {
        cacheBn.reset(mRuntime.first->onCreate());
        cacheBackupBn.reset(mBackupRuntime.first->onCreate(&defaultConfig));
    }
    std::shared_ptr<ComputeCache> packedCache(new ComputeCache(cacheBn, cacheBackupBn));
    packedCache->mCompilerType = mRuntime.first->onGetCompilerType();
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
    auto& inputs = expr->inputs();
    auto& req = expr->inside()->mReq.contentNeedContent;
    MNN_ASSERT(inputs.size() == req.size());
    
    // Create Input's Unit / Cache
    for (int i=0; i<inputs.size(); ++i) {
        if (!req[i]) {
            continue;
        }
        //MNN_PRINT("Use %d\n", i);
        auto inputExpr = inputs[i]->expr();
        if (nullptr != inputExpr.first->inside()->mUnit) {
            continue;
        }
        auto inputCache = inputExpr.first->inside()->mCache;
        if (nullptr != inputCache) {
            inputCaches.insert(inputCache);
            continue;
        }
        _visit(inputExpr.first, inputCaches, inputNode);
    }

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
