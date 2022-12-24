//
//  WrapExecution.cpp
//  MNN
//
//  Created by MNN on 2018/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cmath>
#include "core/WrapExecution.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/CPUCast.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"

// #define LOG_VERBOSE

namespace MNN {
bool WrapExecution::needWrap(const Tensor* input, Backend* curBackend) {
    auto curType = curBackend ? curBackend->type() : MNN_FORWARD_CPU;
    if (curType == MNN_FORWARD_NN) {
        return false;
    }
    auto des = TensorUtils::getDescribe(input);
    auto bn = des->backend;
    MNNForwardType type = MNN_FORWARD_CPU;
    int pack = 4;
    int bytes = 4;
    if (nullptr != bn) {
        type = bn->type();
        if (type == MNN_FORWARD_CPU_EXTENSION) {
            auto core = static_cast<CPUBackend*>(bn)->functions();
            pack = core->pack;
            bytes = core->bytes;
        }
    }
    if (type == curType) {
        return false;;
    }
    bool srcCpu = (type == MNN_FORWARD_CPU_EXTENSION || type == MNN_FORWARD_CPU);
    bool dstCpu = ((curType == MNN_FORWARD_CPU_EXTENSION) || (curType == MNN_FORWARD_CPU));
    if (srcCpu && dstCpu) {
        int curBytes = 4, curPack = 4;
        if (curBackend) {
            auto dstCore = static_cast<CPUBackend*>(curBackend)->functions();
            curBytes = dstCore->bytes;
            curPack = dstCore->pack;
        }
        if (curBytes == bytes) {
            if (curPack == pack || des->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                return false;
            }
        }
    }
    return true;
}

WrapExecution::WrapExecution(Backend* CPUBackend, std::shared_ptr<Execution> execution, bool isStatic)
    : Execution(execution->backend()), mCPUBackend(CPUBackend), mExecution(execution) {
    mValid  = execution->valid();
    mStatic = isStatic;
}
Tensor* WrapExecution::copyConstCache(Tensor* t, Backend* curBackend, std::map<Tensor*, std::shared_ptr<Tensor>>& cache) {
    auto des = TensorUtils::getDescribe(t);
    if ((!des->isMutable) && curBackend->type() != MNN_FORWARD_CPU) {
        auto constCacheiter = cache.find(t);
        if (constCacheiter != cache.end()) {
            // The tensor has been copy by op before, just use it
            return constCacheiter->second.get();
        } else {
            // search or create const for new backend
            std::shared_ptr<Tensor> wrapTensor(new Tensor);
            TensorUtils::copyShape(t, wrapTensor.get(), true);
            wrapTensor->buffer().type = t->buffer().type;
            TensorUtils::adjustTensorForCompability(wrapTensor.get());
            TensorUtils::getDescribe(wrapTensor.get())->quantAttr = TensorUtils::getDescribe(t)->quantAttr;
            TensorUtils::getDescribe(wrapTensor.get())->usage = Tensor::InsideDescribe::CONSTANT;
            auto tempRes = curBackend->onAcquireBuffer(wrapTensor.get(), Backend::STATIC);
            if (!tempRes) {
                return nullptr;
            }
            TensorUtils::getDescribe(wrapTensor.get())->backend = curBackend;
            curBackend->onCopyBuffer(t, wrapTensor.get());
            cache.insert(std::make_pair(t, wrapTensor));
            return wrapTensor.get();
        }
    }
    return nullptr;
}

Tensor* WrapExecution::_getCopyTensor(Tensor* inputTensor, Tensor* outsideInput) {
    auto dstBackend = mExecution->backend();
    auto inputDes   = TensorUtils::getDescribe(inputTensor);
    auto srcBackend = inputDes->backend;

    if (nullptr == srcBackend) {
        srcBackend = mCPUBackend;
    }
    MNN_ASSERT(inputDes->memoryType != Tensor::InsideDescribe::MEMORY_VIRTUAL);
    // CPU -> CPU or XPU -> XPU
    //if (srcBackend == dstBackend) {
    if (srcBackend->type() == dstBackend->type()) {
        return inputTensor;
    }

    auto iter = mInputMaps.find(inputTensor);
    if (iter != mInputMaps.end()) {
        return std::get<2>(iter->second).get();
    }

    auto tensorAddr = inputTensor->host<void*>();
    auto tensorSize = inputTensor->size();

    // CPU -> XPU
    if (srcBackend->type() == mCPUBackend->type()) {

        std::shared_ptr<Tensor> wrapTensor(new Tensor);
        TensorUtils::copyShape(inputTensor, wrapTensor.get(), true);
        TensorUtils::adjustTensorForCompability(wrapTensor.get());
        wrapTensor->buffer().type = inputTensor->buffer().type;
        TensorUtils::getDescribe(wrapTensor.get())->quantAttr = TensorUtils::getDescribe(inputTensor)->quantAttr;
        mInputMaps.insert(std::make_pair(inputTensor, std::make_tuple(dstBackend, dstBackend, wrapTensor)));
#ifdef LOG_VERBOSE
        MNN_PRINT("match cpu to gpu, input:%p, host:%p, wrap:%p, host:%p. dst bn type:%d. outsideInput:%p, refcount:%d\n", inputTensor, inputTensor->host<void*>(), wrapTensor.get(), wrapTensor->host<void*>(), dstBackend->type(), outsideInput, TensorUtils::getDescribe(outsideInput)->useCount);
#endif
        TensorUtils::getDescribe(outsideInput)->useCount++;
        return wrapTensor.get();
    }
    // XPU -> CPU
    if (dstBackend->type() == mCPUBackend->type()) {
        std::shared_ptr<Tensor> wrapTensor(new Tensor);
        TensorUtils::copyShape(inputTensor, wrapTensor.get(), true);
        wrapTensor->buffer().type = inputTensor->buffer().type;
        TensorUtils::adjustTensorForCompability(wrapTensor.get());
        TensorUtils::getDescribe(wrapTensor.get())->quantAttr = TensorUtils::getDescribe(inputTensor)->quantAttr;
        mInputMaps.insert(std::make_pair(inputTensor, std::make_tuple(mCPUBackend, srcBackend, wrapTensor)));
        TensorUtils::getDescribe(outsideInput)->useCount++;
#ifdef LOG_VERBOSE
        MNN_PRINT("match gpu to cpu, input:%p, host:%p, wrap:%p, host:%p. src bn type:%d, outsideInput:%p, refcount:%d\n", inputTensor, inputTensor->host<void*>(), wrapTensor.get(), wrapTensor->host<void*>(), srcBackend->type(), outsideInput, TensorUtils::getDescribe(outsideInput)->useCount);
#endif
        return wrapTensor.get();
    }
    // XPU -> CPU -> XPU'
    std::shared_ptr<Tensor> midTensor(new Tensor);
    std::shared_ptr<Tensor> wrapTensor(new Tensor);
    TensorUtils::copyShape(inputTensor, midTensor.get(), true);
    TensorUtils::copyShape(inputTensor, wrapTensor.get(), true);
    TensorUtils::adjustTensorForCompability(wrapTensor.get());
    TensorUtils::adjustTensorForCompability(midTensor.get());
    TensorUtils::getDescribe(midTensor.get())->usage = TensorUtils::getDescribe(inputTensor)->usage;
    TensorUtils::getDescribe(midTensor.get())->quantAttr = TensorUtils::getDescribe(inputTensor)->quantAttr;
    midTensor->buffer().type                         = inputTensor->buffer().type;
    wrapTensor->buffer().type                        = inputTensor->buffer().type;
    mInputMaps.insert(std::make_pair(inputTensor, std::make_tuple(mCPUBackend, srcBackend, midTensor)));
    mInputMaps.insert(std::make_pair(midTensor.get(), std::make_tuple(dstBackend, dstBackend, wrapTensor)));

    return wrapTensor.get();
}

ErrorCode WrapExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mWrapInputTensors.resize(inputs.size());
    mInputMaps.clear();

    auto dstBackend = mExecution->backend();
    bool isRaster = inputs.size() == 1 && inputs[0] == outputs[0];
    for (int i = 0; i < inputs.size(); ++i) {
        auto inputTensor = inputs[i];
        auto des         = TensorUtils::getDescribe(inputTensor);
        if (isRaster) {
            MNN_ASSERT(inputs.size() == 1);
            mWrapForRaster.reset(new Tensor);
            TensorUtils::copyShape(inputTensor, mWrapForRaster.get(), true);
            mWrapForRaster->buffer().type = inputTensor->buffer().type;
            auto wrapDes                  = TensorUtils::getDescribe(mWrapForRaster.get());
            wrapDes->memoryType           = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            wrapDes->regions              = des->regions;
            for (auto& r : wrapDes->regions) {
                r.origin = _getCopyTensor(r.origin, inputTensor);
            }
            mWrapInputTensors[i] = mWrapForRaster.get();
        } else {
            mWrapInputTensors[i] = _getCopyTensor(inputTensor, inputTensor);
        }
    }


    for (int i = 0; i < outputs.size(); ++i) {
        MNN_ASSERT(TensorUtils::getDescribe(outputs[i])->backend == dstBackend);
    }
    bool memoryAllocSuccess = true;
    // acquire memory, copy const tensors
    for (auto& iter : mInputMaps) {
        auto backend   = std::get<0>(iter.second);
        auto converter = std::get<1>(iter.second);
        auto src       = iter.first;
        auto dst       = std::get<2>(iter.second).get();

        if (TensorUtils::getDescribe(src)->usage == TensorUsage::CONSTANT && mStatic) { // copy constants
            auto srcDes = TensorUtils::getDescribe(src);
            memoryAllocSuccess = backend->onAcquireBuffer(dst, Backend::STATIC);
            if (memoryAllocSuccess) {
                converter->onCopyBuffer(src, dst);
                TensorUtils::getDescribe(dst)->usage = TensorUtils::getDescribe(src)->usage;
            }
        } else {
            memoryAllocSuccess = backend->onAcquireBuffer(dst, Backend::DYNAMIC);
        }
    }
    if (!memoryAllocSuccess) {
        return OUT_OF_MEMORY;
    }

    // do resize
    auto result = mExecution->onResize(mWrapInputTensors, outputs);

    // release memory
    for (auto& iter : mInputMaps) {
        auto backend = std::get<0>(iter.second);
        auto dst     = std::get<2>(iter.second).get();

        if (TensorUtils::getDescribe(dst)->usage == TensorUsage::CONSTANT && mStatic) {
            // Do nothing
        } else {
            backend->onReleaseBuffer(dst, Backend::DYNAMIC);
        }
    }

    return result;
}

ErrorCode WrapExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(mWrapInputTensors.size() == inputs.size());

    // copy variant tensors
    for (auto& iter : mInputMaps) {
        auto converter = std::get<1>(iter.second);
        auto src       = iter.first;
        auto dst       = std::get<2>(iter.second).get();
        if (TensorUtils::getDescribe(src)->usage != TensorUsage::CONSTANT || (!mStatic)) {
            converter->onCopyBuffer(src, dst);
        }
    }
    auto code = mExecution->onExecute(mWrapInputTensors, outputs);
    return code;
}

CastWrapExecution::CastWrapExecution(const CPUBackend::Creator* creator, const Op* op, Backend* backend,
                                     const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, DataType runT)
                                   : Execution(backend), mRunType(runT), mCreator(creator), mType(op->type()), mInputs(inputs) {
    mExecution.reset(mCreator->onCreate(inputs, outputs, op, backend));
}
ErrorCode CastWrapExecution::onResize(const std::vector<Tensor*>& inputs,
                                      const std::vector<Tensor*>& outputs) {
    mWrapInputs.clear();
    mCasts.clear();
    mWrapInputTensor.clear();
    auto& cachedCastTensor = static_cast<CPUBackend*>(backend())->getCachedCastTensor();
    std::vector<Tensor*> realInput;
    if (mType == OpType_Raster) {
        for (const auto& r : TensorUtils::getDescribe(inputs[0])->regions) {
            realInput.push_back(r.origin);
        }
    } else {
        realInput = inputs;
    }
    for (int i = 0; i < realInput.size(); i++) {
        auto input = realInput[i];
        if (CPUBackend::getDataType(input) == mRunType || !OpCommonUtils::opNeedContent(mType, i)) {
            mWrapInputs.push_back(input);
            continue;
        }
        if (cachedCastTensor.find(input) != cachedCastTensor.end()) {
            mWrapInputs.push_back(const_cast<Tensor*>(cachedCastTensor[input]));
            continue;
        }
        std::unique_ptr<Tensor> wrapTensor(new Tensor);
        TensorUtils::copyShape(input, wrapTensor.get(), true);
        TensorUtils::setLinearLayout(wrapTensor.get());
        auto des = TensorUtils::getDescribe(wrapTensor.get());
        auto originDes = TensorUtils::getDescribe(input);
        if (originDes->quantAttr != nullptr) {
            des->quantAttr.reset(new QuantAttr);
            *des->quantAttr = *originDes->quantAttr;
            des->type = mRunType;
        }
        bool memoryAllocSuccess = backend()->onAcquireBuffer(wrapTensor.get(), Backend::DYNAMIC);
        if (!memoryAllocSuccess) {
            return {};
        }
        mWrapInputs.push_back(wrapTensor.get());
        auto wrapPointer = wrapTensor.get();
        mCasts.insert(std::make_pair(input, wrapTensor.get()));
        TensorUtils::getDescribe(wrapPointer)->useCount = TensorUtils::getDescribe(input)->useCount;
        cachedCastTensor.insert(std::make_pair(input, wrapTensor.get()));
        mWrapInputTensor.emplace_back(std::move(wrapTensor));
    }
    ErrorCode res = NO_ERROR;
    if (mType == OpType_Raster) {
        mRasterInputTensor.reset(new Tensor(inputs[0], inputs[0]->getDimensionType(), false));
        mRasterInput = mRasterInputTensor.get();
        TensorUtils::getDescribe(mRasterInput)->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        TensorUtils::getDescribe(mRasterInput)->regions.resize(realInput.size());
        for (int i = 0; i < realInput.size(); i++) {
            TensorUtils::getDescribe(mRasterInput)->regions[i] = TensorUtils::getDescribe(inputs[0])->regions[i];
            TensorUtils::getDescribe(mRasterInput)->regions[i].origin = mWrapInputs[i];
        }
        res = mExecution->onResize({mRasterInput}, outputs);
    } else {
        res = mExecution->onResize(mWrapInputs, outputs);
    }
    for (auto input : realInput) {
        auto iter = cachedCastTensor.find(input);
        if (iter != cachedCastTensor.end()) {
            if (--TensorUtils::getDescribe(iter->second)->useCount == 0) {
                backend()->onReleaseBuffer(iter->second, Backend::DYNAMIC);
            }
        }
    }
    return res;
}
ErrorCode CastWrapExecution::onExecute(const std::vector<Tensor*>& inputs,
                                       const std::vector<Tensor*>& outputs) {
    auto convertType = mRunType == DataType_DT_INT8 ? CPUCastCreator::FlOAT_TO_INT8 : CPUCastCreator::INT8_TO_FlOAT;
    for (const auto& iter : mCasts) {
        auto input = iter.first;
        auto output = iter.second;
        auto cpuBackend = ((CPUBackend*)backend());
        CPUCastCreator::cast(input, output, cpuBackend, convertType);
    }
    if (mType == OpType_Raster) {
        return mExecution->onExecute({ mRasterInput }, outputs);
    } else {
        return mExecution->onExecute(mWrapInputs, outputs);
    }
}
bool CastWrapExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (dst == nullptr || bn == nullptr) {
        return true;
    }
    Execution* exe;
    mExecution->onClone(bn, op, &exe);
    *dst = new CastWrapExecution(bn, mRunType, op, exe);
    return true;
}
} // namespace MNN
