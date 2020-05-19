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
#ifdef MNN_EXPR_ENABLE_PROFILER
#define MNN_EXPRESS_ERROR_REPORT
#endif
#define MNN_EXPRESS_OPEN_MEMORY_REUSE
namespace MNN {
namespace Express {
class Executor::Profiler {
public:
    void reset();
    void dump() const;
    void add(int opType, float timeInMs);
private:
    std::map<int, float> mTimes;
};
void Executor::Profiler::reset() {
    mTimes.clear();
}
void Executor::Profiler::dump() const {
    for (auto iter : mTimes) {
        MNN_PRINT("%s: %f ms\n", EnumNameOpType((OpType)iter.first), iter.second);
    }
}
void Executor::Profiler::add(int opType, float timeInMs) {
    auto iter = mTimes.find(opType);
    if (iter == mTimes.end()) {
        mTimes[opType] = timeInMs;
        return;
    }
    iter->second += timeInMs;
}

void Executor::setGlobalExecutorConfig(MNNForwardType type, const BackendConfig& config, int numberThread) {
    std::lock_guard<std::mutex> _l(mMutex);
    auto creator = MNNGetExtraBackendCreator(type);
    if (nullptr == creator) {
        MNN_ERROR("Error to find creator of %d\n", type);
        return;
    }
    _resetCache();
    Backend::Info info;
    info.type = type;
    info.numThread = numberThread;
    std::shared_ptr<Backend> bn(creator->onCreate(info));
    mBackend = bn;
}
void Executor::_resetCache() {
}

void Executor::gc(GCFlag flag) {
    std::lock_guard<std::mutex> _l(mMutex);
    _resetCache();
    if (FULL == flag) {
        mBackend->onClearBuffer();
        mBackupBackend->onClearBuffer();
    }
}
Executor::Executor(std::shared_ptr<Backend> backend) {
    mBackend = backend;
    if (mBackend->type() == MNN_FORWARD_CPU) {
        mBackupBackend = mBackend;
    } else {
        Backend::Info info;
        info.type = MNN_FORWARD_CPU;
        info.numThread = 1;
        auto creator = MNNGetExtraBackendCreator(MNN_FORWARD_CPU);
        mBackupBackend.reset(creator->onCreate(info));
    }
    _resetCache();
#ifdef MNN_EXPR_ENABLE_PROFILER
    mProfiler.reset(new Profiler);
#endif
}
Executor::~Executor(){
    mBackend = nullptr;
    mBackupBackend = nullptr;
}
void Executor::_addToCache(const std::vector<std::shared_ptr<ComputeCache>>& caches) {
    //FUNC_PRINT(mCaches.size());
}

Executor::Requirement Executor::getRequirement(Expr* expr) const {
    Executor::Requirement req;
    auto op = expr->get();
    auto inputSize = expr->inputs().size();
    req.contentNeedContent.resize(inputSize);
    req.shapeNeedContent.resize(inputSize);
    req.supportError.resize(inputSize);
    if (op->type() == OpType_Extra) {
        for (int i = 0; i < inputSize; ++i) {
            req.contentNeedContent[i] = true;
            req.shapeNeedContent[i]   = false;
            req.supportError[i] = false;
        }
        return req;
    }
    for (int i = 0; i < inputSize; ++i) {
        req.contentNeedContent[i] = SizeComputer::opNeedContent(op->type(), i);
        req.shapeNeedContent[i]   = false;
        if (op->type() != OpType_Concat) {
            req.supportError[i] = false;
        } else {
            req.supportError[i] = true;
        }
    }
    auto needIndexId = SizeComputer::needInputContent(op);
    for (auto index : needIndexId) {
        if (index < req.shapeNeedContent.size()) {
            req.shapeNeedContent[index] = true;
        }
    }
    return req;
}

std::shared_ptr<Executor> Executor::getGlobalExecutor() {
    static std::once_flag of;
    static std::shared_ptr<Executor> gExecutor;
    std::call_once(of, [&]() {
        auto creator = MNNGetExtraBackendCreator(MNN_FORWARD_CPU);
        SizeComputerSuite::init();
        Backend::Info info;
        info.type = MNN_FORWARD_CPU;
        info.numThread = 1;
        std::shared_ptr<Backend> bn(creator->onCreate(info));
        gExecutor.reset(new Executor(bn));
    });
    return gExecutor;
}

ErrorCode Executor::computeInfo(Expr* expr) {
    MNN_ASSERT(nullptr != expr);
    MNN_ASSERT(nullptr != expr->get());
    if (expr->get()->type() == OpType_Extra) {
        return NOT_SUPPORT;
    }
    std::lock_guard<std::mutex> _l(mMutex);
    mStackInputs.resize(expr->inputs().size());
    mStackOutputs.resize(expr->outputSize());
    if (mStack.size() < mStackInputs.size() + mStackOutputs.size()) {
        int origin = (int)mStack.size();
        int destSize = (int)(mStackInputs.size() + mStackOutputs.size());
        for (int i=origin; i<destSize; ++i) {
            mStack.emplace_back(std::shared_ptr<Tensor>(new Tensor));
        }
    }
    for (int i=0; i<mStackInputs.size(); ++i) {
        mStackInputs[i] = mStack[i].get();
    }
    for (int i=0; i<mStackOutputs.size(); ++i) {
        mStackOutputs[i] = mStack[i+(int)mStackInputs.size()].get();
    }
    auto op = expr->get();
    for (int i = 0; i < expr->inputs().size(); ++i) {
        auto inputExpr = expr->inputs()[i]->expr();
        Utils::copyInfoToTensor(mStackInputs[i], inputExpr.first->outputInfo(inputExpr.second));
    }
    bool res = SizeComputer::computeOutputSize(op, mStackInputs, mStackOutputs);
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
    for (int i = 0; i < mStackOutputs.size(); ++i) {
        auto tensor = mStackOutputs[i];
        for (int j = 0; j < tensor->dimensions(); ++j) {
            if (tensor->length(j) <= 0) {
#ifdef MNN_EXPRESS_ERROR_REPORT
                if (nullptr != op->name()) {
                    auto name = op->name()->str();
                    MNN_ERROR("Error to compute shape for %s\n", op->name()->c_str());
                } else {
                    MNN_ERROR("Error to compute shape for %s\n", EnumNameOpType(op->type()));
                }
#endif
                return COMPUTE_SIZE_ERROR;
            }
        }
        auto shape  = expr->outputInfo(i);
        Utils::copyTensorToInfo(shape, tensor);
    }
    return NO_ERROR;
}

void Executor::ComputeCache::syncInput(int offset, const Variable::Info* info) {
    auto tensor = this->getTensor(offset, true);
    Utils::copyInfoToTensor(tensor, info);
}
void Executor::ComputeCache::syncOutput(int offset, Variable::Info* info) {
    auto tensor = this->getTensor(offset, true);
    if (nullptr != tensor) {
        info->ptr = tensor->host<void>();
    }
}

void Executor::ComputeCache::setShapeDirty(int offset, Variable::Info* info) {
    _setShapeDirty();
    if (nullptr != info) {
        syncInput(offset, info);
    }
}

void Executor::ComputeCache::_setShapeDirty() {
    mShapeDirty = true;
}
void Executor::ComputeCache::setContentReady() {
    mContentDirty = false;
}

void Executor::ComputeCache::setContentDirty() {
    mContentDirty = true;
}

void Executor::ComputeCache::TensorContent::reset() {
    auto des = TensorUtils::getDescribe(tensor.get());
    if (nullptr != des->backend && des->useCount >= 0) {
        des->backend->onReleaseBuffer(tensor.get(), Backend::DYNAMIC);
    }
    des->backend = nullptr;
    des->useCount = refCount;
}

class InputCache : public Executor::ComputeCache {
public:
    InputCache() {}
    ~ InputCache() {}
    virtual ErrorCode compute() override {
        if (mContentDirty) {
            return CALL_BACK_STOP;
        }
        return NO_ERROR;
    }
    virtual ErrorCode resize() override {
        return NO_ERROR;
    }
    virtual Tensor* getTensor(int offset, bool host) override {
        return &mTensor;
    }

private:
    Tensor mTensor;
};
class PipelineCache : public Executor::ComputeCache {
public:
    PipelineCache();
    virtual ~ PipelineCache();
    virtual Tensor* getTensor(int offset, bool host) override {
        auto tensor = mOutputs[offset];
        if (tensor->host<void>() != nullptr || !host) {
            return tensor;
        }
        auto iter = mCopyOutputs.find(tensor);
        if (iter == mCopyOutputs.end()) {
            // First get tensor, create and copy
            TensorContent content;
            content.tensor.reset(new Tensor);
            TensorUtils::copyShape(tensor, content.tensor.get(), true);
            bool res = mBackupBackend->onAcquireBuffer(content.tensor.get(), Backend::DYNAMIC);
            if (!res) {
                MNN_ERROR("Malloc error when copy out\n");
                return nullptr;
            }
            tensor->copyToHostTensor(content.tensor.get());
            mCopyOutputs.insert(std::make_pair(tensor, content.tensor.get()));
            mTensors.emplace_back(std::move(content));
            iter = mCopyOutputs.find(tensor);
        }
        return iter->second;
    }
    virtual ErrorCode compute() override;
    virtual ErrorCode resize() override;
private:
    std::set<std::shared_ptr<ComputeCache>> mInputs;
    std::vector<Tensor*> mOutputs;
    std::vector<TensorContent> mTensors;
    std::vector<std::shared_ptr<Unit>> mUnits;
    std::map<Tensor*, Tensor*> mCopyOutputs;
    std::shared_ptr<Backend> mBackend;
    std::shared_ptr<Backend> mBackupBackend;
    friend class Executor;
};

struct Executor::ComputeCache::Unit {
    std::vector<Tensor*> inputs;
    std::vector<int> inputsNeedRelease;
    std::vector<Tensor*> outputs;
    const Op* op;
    std::weak_ptr<Expr::Inside> inside;
    std::shared_ptr<Execution> exe;
    std::shared_ptr<char> extraBuffer;
    std::vector<std::pair<Tensor*, const Variable::Info*>> inputOutsides;
};
PipelineCache::PipelineCache() {
    // Do nothing
}

PipelineCache::~PipelineCache() {
    mUnits.clear();
    for (auto t : mTensors) {
        t.reset();
    }
}
ErrorCode PipelineCache::compute() {
    if (mShapeDirty) {
        auto code = resize();
        if (NO_ERROR != code) {
            return code;
        }
    }
    if (!mContentDirty) {
        return NO_ERROR;
    }
    for (auto c : mInputs) {
        auto code = c->compute();
        if (NO_ERROR != code) {
            return code;
        }
    }
    mBackend->onExecuteBegin();
    //mBackupBackend->onExecuteBegin();
    for (int i=0; i<mUnits.size(); ++i) {
        auto& iter = *mUnits[i];
        if (nullptr == iter.exe) {
            continue;
        }
        auto inside = iter.inside.lock();
        if (nullptr == inside || inside->mInfoDirty) {
            continue;
        }
#ifdef MNN_EXPR_ENABLE_PROFILER
        Timer autoTime;
#endif
        auto code = iter.exe->onExecute(iter.inputs, iter.outputs);
#ifdef MNN_EXPR_ENABLE_PROFILER
        float costTime = (float)autoTime.durationInUs() / (float)1000;
        Executor::getGlobalExecutor()->addOpCostTime((int)mUnits[i]->op->type(), costTime);
#endif
        if (NO_ERROR != code) {
#ifdef MNN_EXPRESS_ERROR_REPORT
            MNN_ERROR("Error to compute shape for %s\n", EnumNameOpType(iter.op->type()));
#endif
            mBackend->onExecuteEnd();
            return code;
        }
        inside->mContentDirty = false;
    }
    mBackend->onExecuteEnd();
    //mBackupBackend->onExecuteEnd();
    for (auto iter : mCopyOutputs) {
        iter.first->copyToHostTensor(iter.second);
    }
    mContentDirty = false;
    return NO_ERROR;
}

ErrorCode PipelineCache::resize() {
    if (!mShapeDirty) {
        return NO_ERROR;
    }
    for (auto c : mInputs) {
        auto code = c->resize();
        if (NO_ERROR != code) {
            return code;
        }
    }
    for (auto& t : mTensors) {
        t.reset();
    }
    for (auto& tensor : mOutputs) {
        TensorUtils::getDescribe(tensor)->useCount += 1;
    }
    mShapeDirty = false;
    for (int unitIndex = 0; unitIndex < mUnits.size(); ++unitIndex) {
        auto& iter = *mUnits[unitIndex];
        auto inside = iter.inside.lock();
        if (nullptr == inside || inside->mInfoDirty) {
            mShapeDirty = true;
            continue;
        }
        for (auto& tensor : iter.inputOutsides) {
            Utils::copyInfoToTensor(tensor.first, tensor.second);
        }
        for (int i=0; i<iter.outputs.size(); ++i) {
            Utils::copyInfoToTensor(iter.outputs[i], inside->mOutputInfos.data() + i);
            iter.outputs[i]->buffer().host = nullptr;
        }
        if (nullptr == iter.exe) {
#ifdef MNN_EXPR_ENABLE_PROFILER
            Timer autoTime;
#endif
            iter.exe.reset(mBackend->onCreate(iter.inputs, iter.outputs, iter.op));
            if (nullptr == iter.exe) {
                iter.exe.reset(mBackupBackend->onCreate(iter.inputs, iter.outputs, iter.op));
            }
            // Check if need wrap
            bool needWrap = false;
            auto bn = iter.exe->backend();
            auto iterType = bn->type();
            for (int i=0; i<inside->mReq.contentNeedContent.size(); ++i) {
                if (!inside->mReq.contentNeedContent[i]) {
                    continue;
                }
                auto tensorBn = TensorUtils::getDescribe(iter.inputs[i])->backend;
                auto type = MNN_FORWARD_CPU;
                if (nullptr != tensorBn) {
                    type = tensorBn->type();
                }
                if (iterType != type) {
                    needWrap = true;
                    break;
                }
            }
            if (needWrap) {
                iter.exe.reset(new WrapExecution(mBackupBackend.get(), iter.exe));
            }

#ifdef MNN_EXPR_ENABLE_PROFILER
            float costTime = (float)autoTime.durationInUs() / (float)1000;
            Executor::getGlobalExecutor()->addOpCostTime((int)iter.op->type(), costTime);
#endif
        }
        if (nullptr == iter.exe) {
            return NOT_SUPPORT;
        }
#ifdef MNN_EXPR_ENABLE_PROFILER
        Timer autoTime;
#endif
        auto bn = iter.exe->backend();
        for (int i=0; i<iter.outputs.size(); ++i) {
            auto res = bn->onAcquireBuffer(iter.outputs[i], Backend::DYNAMIC);
            TensorUtils::getDescribe(iter.outputs[i])->backend = bn;
            if (!res) {
                return OUT_OF_MEMORY;
            }
        }
        auto code= iter.exe->onResize(iter.inputs, iter.outputs);
#ifdef MNN_EXPR_ENABLE_PROFILER
        float costTime = (float)autoTime.durationInUs() / (float)1000;
        Executor::getGlobalExecutor()->addOpCostTime((int)iter.op->type(), costTime);
#endif
        if (NO_ERROR != code) {
            return code;
        }
#ifdef MNN_EXPRESS_OPEN_MEMORY_REUSE
        for (int i=0; i<iter.inputsNeedRelease.size(); ++i) {
            auto index = iter.inputsNeedRelease[i];
            auto des = TensorUtils::getDescribe(iter.inputs[index]);
            des->useCount--;
            if (des->useCount <= 0 && des->backend != nullptr) {
                des->backend->onReleaseBuffer(iter.inputs[index], Backend::DYNAMIC);
                //Set useCount < 0, so tensorContent's reset will not release it
                des->useCount = -1;
            }
        }
#endif
    }
    for (auto iter : mCopyOutputs) {
        TensorUtils::copyShape(iter.first, iter.second, true);
        bool res = mBackupBackend->onAcquireBuffer(iter.second, Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
    }
    mContentDirty = true;
    return NO_ERROR;
}

static void _collectExecuteUnit(std::vector<std::shared_ptr<Executor::ComputeCache::Unit>>& dest, EXPRP expr) {
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
    expr->inside()->mLinkCache = true;
    dest.emplace_back(std::move(unit));
    expr->inside()->mUnit = nullptr;
}

void Executor::_createSingle(EXPRP expr) {
    MNN_ASSERT(expr->get() == nullptr);
    auto cache = expr->inside()->mCache;
    cache.reset(new InputCache);
    expr->inside()->mCache = cache;
    expr->inside()->mCacheOffset = 0;
    cache->syncInput(0, expr->outputInfo(0));
    if (VARP::INPUT == expr->inputType()) {
        cache->setContentDirty();
    } else {
        cache->setContentReady();
    }
}

void Executor::_create(const std::vector<EXPRP>& outputs, std::set<std::shared_ptr<Executor::ComputeCache>>&& inputCaches, std::vector<ComputeCache::TensorContent>&& tensors, bool forceCPU) {
    std::vector<EXPRP> packed;
    for (auto expr : outputs) {
        // Make Cache For Single Tensor
        auto cache = expr->inside()->mCache;
        if (nullptr != cache) {
            continue;
        }
        if (nullptr != expr->get()) {
            packed.emplace_back(expr);
            continue;
        }
        _createSingle(expr);
    }
    if (packed.empty()) {
        return;
    }
    std::shared_ptr<PipelineCache> packedCache(new PipelineCache);
    if (forceCPU) {
        packedCache->mBackend = mBackupBackend;
    } else {
        packedCache->mBackend = mBackend;
    }
    packedCache->mInputs = std::move(inputCaches);
    for (auto expr : packed) {
        expr->inside()->mCacheOffset = (int)packedCache->mOutputs.size();
        MNN_ASSERT(expr->inside()->mUnit != nullptr);
        auto& originOutputs = expr->inside()->mUnit->outputs;
        for (auto t : originOutputs) {
            packedCache->mOutputs.emplace_back(t);
        }
        expr->inside()->mCache = std::static_pointer_cast<ComputeCache>(packedCache);
    }
    packedCache->mTensors = std::move(tensors);
    packedCache->mBackupBackend = mBackupBackend;
    
    // Backup Tensor Refcount
    for (auto& t : packedCache->mTensors) {
        t.refCount = TensorUtils::getDescribe(t.tensor.get())->useCount;
    }
    // Create Units
    for (auto expr : packed) {
        _collectExecuteUnit(packedCache->mUnits, expr);
    }
}

void Executor::_visit(EXPRP expr, std::set<std::shared_ptr<Executor::ComputeCache>>& inputCaches, std::vector<ComputeCache::TensorContent>& tensors) {
    auto& inputs = expr->inputs();
    auto& req = expr->inside()->mReq.contentNeedContent;
    MNN_ASSERT(inputs.size() == req.size());
    
    // Create Input's Unit / Cache
    for (int i=0; i<inputs.size(); ++i) {
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
        _visit(inputExpr.first, inputCaches, tensors);
    }

    // Create Self Unit / Cache
    auto op = expr->get();
    if (nullptr == op) {
        // Make Cache For Single Tensor
        _createSingle(expr);
        inputCaches.insert(expr->inside()->mCache);
        return;
    }
    std::shared_ptr<ComputeCache::Unit> unitP(new ComputeCache::Unit);
    ComputeCache::Unit& unit = *unitP;
    unit.op = expr->get();
    unit.extraBuffer = expr->extra().first;
    unit.inside = std::weak_ptr<Expr::Inside>(expr->inside());
    unit.inputs.resize(inputs.size());
    for (int i=0; i<inputs.size(); ++i) {
        auto inputExpr = inputs[i]->expr();
        if (!req[i]) {
            // The compute don't need it, but need shape info for exe's onResize
            ComputeCache::TensorContent content;
            content.tensor.reset(new Tensor);
            unit.inputOutsides.emplace_back(std::make_pair(content.tensor.get(), inputExpr.first->outputInfo(inputExpr.second)));
            unit.inputs[i] = content.tensor.get();
            tensors.emplace_back(std::move(content));
            continue;
        }
        auto inputUnit = inputExpr.first->inside()->mUnit;
        if (nullptr != inputUnit) {
            unit.inputs[i] = inputUnit->outputs[inputExpr.second];
            TensorUtils::getDescribe(unit.inputs[i])->useCount++;
            unit.inputsNeedRelease.emplace_back(i);
            continue;
        }
        auto inputCache = inputExpr.first->inside()->mCache;
        if (nullptr != inputCache) {
            unit.inputs[i] = inputCache->getTensor(inputExpr.first->inside()->mCacheOffset + inputExpr.second, false);
            continue;
        }
        MNN_ASSERT(false);
    }
    unit.outputs.resize(expr->outputSize());
    for (int i=0; i<unit.outputs.size(); ++i) {
        ComputeCache::TensorContent content;
        content.tensor.reset(new Tensor);
        unit.outputs[i] = content.tensor.get();
        tensors.emplace_back(std::move(content));
    }
    expr->inside()->mUnit = unitP;
}

void Executor::makeCache(const std::vector<EXPRP>& expr, bool forceCPU) {
    std::lock_guard<std::mutex> _l(mMutex);
    //FUNC_PRINT(mCaches.size());
    std::set<std::shared_ptr<Executor::ComputeCache>> inputCaches;
    std::vector<ComputeCache::TensorContent> tensors;
    for (auto e : expr) {
        _visit(e, inputCaches, tensors);
    }
    _create(expr, std::move(inputCaches), std::move(tensors), forceCPU);
}
void Executor::addOpCostTime(int op, float costTime) {
#ifdef MNN_EXPR_ENABLE_PROFILER
    mProfiler->add(op, costTime);
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
