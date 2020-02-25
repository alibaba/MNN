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
#include "core/Backend.hpp"
#include <MNN/Tensor.hpp>
#include "core/TensorUtils.hpp"
#include <MNN/AutoTime.hpp>
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
    mInputs.resize(expr->inputs().size());
    mOutputs.resize(expr->outputSize());
    if (mStack.size() < mInputs.size() + mOutputs.size()) {
        int origin = (int)mStack.size();
        int destSize = (int)(mInputs.size() + mOutputs.size());
        for (int i=origin; i<destSize; ++i) {
            mStack.emplace_back(std::shared_ptr<Tensor>(new Tensor));
        }
    }
    for (int i=0; i<mInputs.size(); ++i) {
        mInputs[i] = mStack[i].get();
    }
    for (int i=0; i<mOutputs.size(); ++i) {
        mOutputs[i] = mStack[i+(int)mInputs.size()].get();
    }
    auto op = expr->get();
    for (int i = 0; i < expr->inputs().size(); ++i) {
        auto inputExpr = expr->inputs()[i]->expr();
        Utils::copyInfoToTensor(mInputs[i], inputExpr.first->outputInfo(inputExpr.second));
    }
    bool res = SizeComputer::computeOutputSize(op, mInputs, mOutputs);
    if (!res) {
        // Compute Error
#ifdef MNN_EXPRESS_ERROR_REPORT
        FUNC_PRINT(op->type());
#endif
        return COMPUTE_SIZE_ERROR;
    }
    for (int i = 0; i < mOutputs.size(); ++i) {
        auto tensor = mOutputs[i];
        for (int j = 0; j < tensor->dimensions(); ++j) {
            if (tensor->length(j) <= 0) {
#ifdef MNN_EXPRESS_ERROR_REPORT
                if (nullptr != op->name()) {
                    auto name = op->name()->str();
                    MNN_ERROR("Error to compute shape for %s\n", op->name()->c_str());
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

Executor::ComputeCache::~ComputeCache() {
    mUnits.clear();
    for (auto t : mTensors) {
        t.reset();
    }
}

void Executor::ComputeCache::setShapeDirty() {
    mShapeDirty = true;
    for (auto iter : mLinks) {
        auto cache = iter.lock();
        if (nullptr != cache && false == cache->mShapeDirty) {
            cache->setShapeDirty();
        }
    }
}
void Executor::ComputeCache::setContentDirty() {
    mContentDirty = true;
    for (auto iter : mLinks) {
        auto cache = iter.lock();
        if (nullptr != cache && false == cache->mContentDirty) {
            cache->setContentDirty();
        }
    }
}

void Executor::ComputeCache::TensorContent::reset() {
    auto des = TensorUtils::getDescribe(tensor.get());
    des->useCount = refCount;
    if (nullptr != des->backend) {
        des->backend->onReleaseBuffer(tensor.get(), Backend::DYNAMIC);
        des->backend = nullptr;
    }
}
void Executor::ComputeCache::addLink(std::shared_ptr<ComputeCache> cache) {
    for (int i=0; i<mLinks.size(); ++i) {
        auto ptr = mLinks[i].lock().get();
        if (ptr == cache.get()) {
            return;
        }
        if (ptr == nullptr) {
            mLinks[i] = std::weak_ptr<ComputeCache>(cache);
            return;
        }
    }
    mLinks.emplace_back(std::weak_ptr<ComputeCache>(cache));
}
Tensor* Executor::ComputeCache::output(EXPRP outputExpr, int index, bool host) const {
    auto iter = mOutputTensors.find(outputExpr.get());
    if (iter == mOutputTensors.end()) {
        return nullptr;
    }
    MNN_ASSERT(index >= 0 && index < iter->second.size());
    if (host) {
        return iter->second[index].first;
    }
    return iter->second[index].second;
}
void Executor::ComputeCache::dup(EXPRP src, EXPRP dst) {
    if (mOutputTensors.find(src.get()) == mOutputTensors.end()) {
        return;
    }
    mOutputTensors[dst.get()] = mOutputTensors[src.get()];
}
void Executor::ComputeCache::recycle(Expr* expr) {
    mOutputTensors.erase(expr);
    if (mOutputTensors.empty()) {
        mUnits.clear();
        for (auto& t : mTensors) {
            t.reset();
        }
        mTensors.clear();
        mInputs.clear();
    }
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
    for (auto c : mInputs) {
        auto code = c->compute();
        if (NO_ERROR != code) {
            return code;
        }
    }
    mBackend->onExecuteBegin();
    for (int i=0; i<mUnits.size(); ++i) {
        auto& iter = mUnits[i];
        if (nullptr == iter.exe) {
            continue;
        }
        //FUNC_PRINT_ALL(EnumNameOpType(iter.origin->get()->type()), s);
#ifdef MNN_EXPR_ENABLE_PROFILER
        Timer autoTime;
#endif
        auto code = iter.exe->onExecute(iter.inputs, iter.outputs);
#ifdef MNN_EXPR_ENABLE_PROFILER
        float costTime = (float)autoTime.durationInUs() / (float)1000;
        Executor::getGlobalExecutor()->addOpCostTime((int)mUnits[i].origin->get()->type(), costTime);
#endif
        if (NO_ERROR != code) {
            mBackend->onExecuteEnd();
            return code;
        }
    }
    mBackend->onExecuteEnd();
    for (auto& iter : mOutputTensors) {
        for (auto& output : iter.second) {
            TensorUtils::getDescribe(output.second)->useCount = 0;
        }
    }
    for (auto& iter : mOutputTensors) {
        for (auto& output : iter.second) {
            if (TensorUtils::getDescribe(output.second)->useCount > 0) {
                continue;
            }
            if (mUnits.empty()) {
                output.second->copyFromHostTensor(output.first);
            } else {
                output.second->copyToHostTensor(output.first);
            }
            TensorUtils::getDescribe(output.second)->useCount = 1;
        }
    }
    mContentDirty = false;
    return NO_ERROR;
}


ErrorCode Executor::ComputeCache::resize() {
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
    if (mUnits.empty()) {
        // Single Tensor
        auto iter = mOutputTensors.begin();
        auto expr = iter->first;
        Utils::copyInfoToTensor(iter->second[0].first, expr->outputInfo(0));
        iter->second[0].first->buffer().device = 0;
    }
    for (auto& iter : mUnits) {
        if ((iter.origin->infoDirty()) || (!iter.origin->valid())) {
            for (int i=0; i<iter.outputs.size(); ++i) {
                iter.outputs[i]->buffer().dimensions = 0;
            }
            continue;
        }
        for (int i=0; i<iter.outputs.size(); ++i) {
            Utils::copyInfoToTensor(iter.outputs[i], iter.origin->outputInfo(i));
            auto res = mBackend->onAcquireBuffer(iter.outputs[i], Backend::DYNAMIC);
            TensorUtils::getDescribe(iter.outputs[i])->backend = mBackend.get();
            if (!res) {
                return OUT_OF_MEMORY;
            }
        }
        if (nullptr == iter.exe) {
#ifdef MNN_EXPR_ENABLE_PROFILER
            Timer autoTime;
#endif
            iter.exe.reset(mBackend->onCreate(iter.inputs, iter.outputs, iter.origin->get()));
#ifdef MNN_EXPR_ENABLE_PROFILER
            float costTime = (float)autoTime.durationInUs() / (float)1000;
            Executor::getGlobalExecutor()->addOpCostTime((int)iter.origin->get()->type(), costTime);
#endif
        }
        if (nullptr == iter.exe) {
            return NOT_SUPPORT;
        }
#ifdef MNN_EXPR_ENABLE_PROFILER
        Timer autoTime;
#endif
        auto code= iter.exe->onResize(iter.inputs, iter.outputs);
#ifdef MNN_EXPR_ENABLE_PROFILER
        float costTime = (float)autoTime.durationInUs() / (float)1000;
        Executor::getGlobalExecutor()->addOpCostTime((int)iter.origin->get()->type(), costTime);
#endif
        if (NO_ERROR != code) {
            return code;
        }
        auto& req = iter.origin->inside()->mReq.contentNeedContent;
        for (int i=0; i<iter.inputs.size(); ++i) {
            if (iter.inputFromCache[i]) {
                continue;
            }
            if (!req[i]) {
                continue;
            }
            auto des = TensorUtils::getDescribe(iter.inputs[i]);
            des->useCount--;
            if (des->useCount <= 0 && des->backend != nullptr) {
                des->backend->onReleaseBuffer(iter.inputs[i], Backend::DYNAMIC);
                des->backend = nullptr;
            }
        }
    }
    for (auto& iter : mOutputTensors) {
        auto expr = iter.first;
        for (int i=0; i<iter.second.size(); ++i) {
            if (mUnits.empty()) {
                // For Single Tensor, Host -> Device
                if (iter.second[i].first != iter.second[i].second) {
                    TensorUtils::copyShape(iter.second[i].first, iter.second[i].second, true);
                    iter.second[i].second->buffer().host = nullptr;
                    auto res = mBackend->onAcquireBuffer(iter.second[i].second, Backend::DYNAMIC);
                    if (!res) {
                        return OUT_OF_MEMORY;
                    }
                    TensorUtils::getDescribe(iter.second[i].second)->backend = mBackend.get();
                }
            } else {
                // For Other Cache, Device -> Host
                if (iter.second[i].first != iter.second[i].second) {
                    TensorUtils::copyShape(iter.second[i].second, iter.second[i].first, true);
                    iter.second[i].first->buffer().device = 0;
                    auto res = mBackupBackend->onAcquireBuffer(iter.second[i].first, Backend::DYNAMIC);
                    if (!res) {
                        return OUT_OF_MEMORY;
                    }
                    TensorUtils::getDescribe(iter.second[i].first)->backend = mBackupBackend.get();
                }
            }
            expr->outputInfo(i)->ptr = iter.second[i].first->host<void>();
        }
    }
    mShapeDirty = false;
    mContentDirty = true;
    return NO_ERROR;
}

static void _collectExecuteUnit(std::vector<Executor::ComputeCache::Unit>& dest, EXPRP expr, std::map<EXPRP, Executor::ComputeCache::Unit>& units) {
    auto& inputs = expr->inputs();
    auto& req = expr->inside()->mReq.contentNeedContent;
    MNN_ASSERT(inputs.size() == req.size());
    
    for (int i=0; i<inputs.size(); ++i) {
        if (!req[i]) {
            continue;
        }
        auto inputExpr = inputs[i]->expr();
        if (units.find(inputExpr.first) == units.end()) {
            continue;
        }
        auto inputCache = inputExpr.first->inside()->mCache;
        if (nullptr != inputCache) {
            continue;
        }
        _collectExecuteUnit(dest, inputExpr.first, units);
    }
    auto iter = units.find(expr);
    if (iter == units.end()) {
        return;
    }
    dest.emplace_back(std::move(iter->second));
    units.erase(iter);
}

void Executor::ComputeCache::create(const std::vector<EXPRP>& outputs, std::map<EXPRP, ComputeCache::Unit>& units, std::set<std::shared_ptr<Executor::ComputeCache>>&& inputCaches, std::vector<ComputeCache::TensorContent>&& tensors, std::shared_ptr<Backend> bn, std::shared_ptr<Backend> backup) {
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
        cache.reset(new ComputeCache);
        cache->mBackend = bn;
        cache->mTensors.resize(1);
        cache->mTensors[0].tensor.reset(new Tensor);
        Utils::copyInfoToTensor(cache->mTensors[0].tensor.get(), expr->outputInfo(0));
        expr->inside()->mCache = cache;
        if (bn->type() != MNN_FORWARD_CPU) {
            cache->mTensors.resize(2);
            cache->mTensors[1].tensor.reset(new Tensor);
            Utils::copyInfoToTensor(cache->mTensors[1].tensor.get(), expr->outputInfo(0));
            cache->mTensors[1].tensor->buffer().host = nullptr;
            cache->mOutputTensors[expr.get()] = {std::make_pair(cache->mTensors[0].tensor.get(), cache->mTensors[1].tensor.get())};
        } else {
            cache->mOutputTensors[expr.get()] = {std::make_pair(cache->mTensors[0].tensor.get(), cache->mTensors[0].tensor.get())};
        }
        cache->mBackupBackend = backup;
    }
    if (packed.empty()) {
        return;
    }
    std::shared_ptr<ComputeCache> packedCache(new ComputeCache);
    packedCache->mBackend = bn;
    packedCache->mInputs = std::move(inputCaches);
    for (auto input : packedCache->mInputs) {
        input->addLink(packedCache);
    }
    for (auto expr : packed) {
        MNN_ASSERT(units.find(expr) != units.end());
        auto& originOutputs = units[expr].outputs;
        std::vector<std::pair<Tensor*, Tensor*>> destOutputs;
        if (bn->type() == MNN_FORWARD_CPU) {
            for (auto t : originOutputs) {
                destOutputs.emplace_back(std::make_pair(t, t));
            }
        } else {
            for (auto t : originOutputs) {
                ComputeCache::TensorContent content;
                content.tensor.reset(new Tensor);
                TensorUtils::copyShape(t, content.tensor.get(), true);
                destOutputs.emplace_back(std::make_pair(content.tensor.get(), t));
                tensors.emplace_back(std::move(content));
            }
        }
        packedCache->mOutputTensors[expr.get()] = std::move(destOutputs);
        expr->inside()->mCache = packedCache;
    }
    packedCache->mTensors = std::move(tensors);
    packedCache->mBackupBackend = backup;
    
    // Backup Tensor Refcount
    for (auto& t : packedCache->mTensors) {
        t.refCount = TensorUtils::getDescribe(t.tensor.get())->useCount;
    }
    // Create Units
    for (auto expr : packed) {
        _collectExecuteUnit(packedCache->mUnits, expr, units);
    }
    // Resize if possible
    packedCache->resize();
}

void Executor::_visit(EXPRP expr, std::map<EXPRP, ComputeCache::Unit>& units, std::set<std::shared_ptr<Executor::ComputeCache>>& inputCaches, std::vector<ComputeCache::TensorContent>& tensors) {
    auto& inputs = expr->inputs();
    auto& req = expr->inside()->mReq.contentNeedContent;
    MNN_ASSERT(inputs.size() == req.size());
    
    // Create Input's Unit / Cache
    for (int i=0; i<inputs.size(); ++i) {
        if (!req[i]) {
            continue;
        }
        auto inputExpr = inputs[i]->expr();
        if (units.find(inputExpr.first) != units.end()) {
            continue;
        }
        auto inputCache = inputExpr.first->inside()->mCache;
        if (nullptr != inputCache) {
            inputCaches.insert(inputCache);
            continue;
        }
        _visit(inputExpr.first, units, inputCaches, tensors);
    }
    
    // Create Self Unit / Cache
    auto op = expr->get();
    if (nullptr == op) {
        // Make Cache For Single Tensor
        Executor::ComputeCache::create({expr}, units, {}, {}, mBackend, mBackupBackend);
        return;
    }
    ComputeCache::Unit unit;
    unit.origin = expr.get();
    unit.inputs.resize(inputs.size());
    unit.inputFromCache.resize(inputs.size());
    for (int i=0; i<inputs.size(); ++i) {
        auto inputExpr = inputs[i]->expr();
        if (!req[i]) {
            ComputeCache::TensorContent content;
            content.tensor.reset(new Tensor);
            Utils::copyInfoToTensor(content.tensor.get(), inputExpr.first->outputInfo(inputExpr.second));
            unit.inputs[i] = content.tensor.get();
            tensors.emplace_back(std::move(content));
            continue;
        }
        auto iter = units.find(inputExpr.first);
        if (iter != units.end()) {
            unit.inputs[i] = iter->second.outputs[inputExpr.second];
            TensorUtils::getDescribe(unit.inputs[i])->useCount++;
            unit.inputFromCache[i] = false;
            continue;
        }
        auto inputCache = inputExpr.first->inside()->mCache;
        if (nullptr != inputCache) {
            unit.inputs[i] = inputCache->output(inputExpr.first, inputExpr.second, false);
            unit.inputFromCache[i] = true;
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
    units.insert(std::make_pair(expr, std::move(unit)));
}

void Executor::makeCache(std::vector<EXPRP> expr) {
    std::lock_guard<std::mutex> _l(mMutex);
    //FUNC_PRINT(mCaches.size());
    std::map<EXPRP, ComputeCache::Unit> units;
    std::set<std::shared_ptr<Executor::ComputeCache>> inputCaches;
    std::vector<ComputeCache::TensorContent> tensors;
    for (auto e : expr) {
        _visit(e, units, inputCaches, tensors);
    }
    Executor::ComputeCache::create(expr, units, std::move(inputCaches), std::move(tensors), mBackend, mBackupBackend);
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
