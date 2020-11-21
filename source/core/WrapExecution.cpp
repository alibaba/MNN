//
//  WrapExecution.cpp
//  MNN
//
//  Created by MNN on 2018/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/WrapExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

WrapExecution::WrapExecution(Backend* CPUBackend, std::shared_ptr<Execution> execution, bool isStatic)
    : Execution(execution->backend()), mCPUBackend(CPUBackend), mExecution(execution) {
    mValid  = execution->valid();
    mStatic = isStatic;
}

Tensor* WrapExecution::_getCopyTensor(Tensor* inputTensor) {
    auto* dstBackend = mExecution->backend();
    auto* inputDes   = TensorUtils::getDescribe(inputTensor);
    auto* srcBackend = inputDes->backend;
    if (nullptr == srcBackend) {
        srcBackend = mCPUBackend;
    }
    // CPU -> CPU or XPU -> XPU
    if (srcBackend == dstBackend) {
        return inputTensor;
    }
    auto iter = mInputMaps.find(inputTensor);
    if (iter != mInputMaps.end()) {
        return std::get<2>(iter->second).get();
    }
    // CPU -> XPU
    if (srcBackend == mCPUBackend) {
        auto* wrapTensor = new Tensor;
        TensorUtils::copyShape(inputTensor, wrapTensor, true);
        wrapTensor->buffer().type = inputTensor->buffer().type;
        mInputMaps.emplace(inputTensor, std::make_tuple(dstBackend, dstBackend, wrapTensor));
        return wrapTensor;
    }
    // XPU -> CPU
    if (dstBackend == mCPUBackend) {
        auto* wrapTensor = new Tensor;
        TensorUtils::copyShape(inputTensor, wrapTensor, true);
        wrapTensor->buffer().type = inputTensor->buffer().type;
        mInputMaps.emplace(inputTensor, std::make_tuple(mCPUBackend, srcBackend, wrapTensor));
        return wrapTensor;
    }
    // XPU -> CPU -> XPU'
    auto* midTensor = new Tensor;
    auto* wrapTensor = new Tensor;
    TensorUtils::copyShape(inputTensor, midTensor, true);
    TensorUtils::copyShape(inputTensor, wrapTensor, true);
    TensorUtils::getDescribe(midTensor)->usage = TensorUtils::getDescribe(inputTensor)->usage;
    midTensor->buffer().type                         = inputTensor->buffer().type;
    wrapTensor->buffer().type                        = inputTensor->buffer().type;
    mInputMaps.emplace(inputTensor, std::make_tuple(mCPUBackend, srcBackend, midTensor));
    mInputMaps.emplace(midTensor, std::make_tuple(dstBackend, dstBackend, wrapTensor));
    return wrapTensor;
}

ErrorCode WrapExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mWrapInputTensors.resize(inputs.size());
    mInputMaps.clear();

    auto* dstBackend = mExecution->backend();
    for (int i = 0; i < inputs.size(); ++i) {
        auto* inputTensor = inputs[i];
        auto* des         = TensorUtils::getDescribe(inputTensor);
        if (des->memoryType == Tensor::InsideDescribe::MemoryType::MEMORY_VIRTUAL) {
            MNN_ASSERT(inputs.size() == 1)
            mWrapForRaster.reset(new Tensor);
            TensorUtils::copyShape(inputTensor, mWrapForRaster.get(), true);
            mWrapForRaster->buffer().type = inputTensor->buffer().type;
            auto* wrapDes                 = TensorUtils::getDescribe(mWrapForRaster.get());
            wrapDes->memoryType           = Tensor::InsideDescribe::MemoryType::MEMORY_VIRTUAL;
            wrapDes->regions              = des->regions;
            for (auto& r : wrapDes->regions) {
                r.origin = _getCopyTensor(r.origin);
            }
            mWrapInputTensors[i] = mWrapForRaster.get();
        } else {
            mWrapInputTensors[i] = _getCopyTensor(inputTensor);
        }
    }

    for (auto* output : outputs) {
        MNN_ASSERT(TensorUtils::getDescribe(output)->backend == dstBackend)
    }
    bool memoryAllocSuccess = true;
    // acquire memory, copy const tensors
    for (auto& iter : mInputMaps) {
        auto* backend   = std::get<0>(iter.second);
        auto* converter = std::get<1>(iter.second);
        auto* src       = iter.first;
        auto* dst       = std::get<2>(iter.second).get();

        if (TensorUtils::getDescribe(src)->usage == TensorUsage::CONSTANT && mStatic) {
            memoryAllocSuccess = backend->onAcquireBuffer(dst, Backend::DYNAMIC_SEPERATE);
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
    const auto result = mExecution->onResize(mWrapInputTensors, outputs);

    // release memory
    for (auto& iter : mInputMaps) {
        auto* backend = std::get<0>(iter.second);
        auto* dst     = std::get<2>(iter.second).get();

        if (TensorUtils::getDescribe(dst)->usage == TensorUsage::CONSTANT && mStatic) {
            backend->onReleaseBuffer(dst, Backend::DYNAMIC_SEPERATE);
        } else {
            backend->onReleaseBuffer(dst, Backend::DYNAMIC);
        }
    }
    return result;
}

ErrorCode WrapExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(mWrapInputTensors.size() == inputs.size())

    // copy variant tensors
    for (auto& iter : mInputMaps) {
        auto* converter = std::get<1>(iter.second);
        auto* src       = iter.first;
        auto* dst       = std::get<2>(iter.second).get();
        if (TensorUtils::getDescribe(src)->usage != TensorUsage::CONSTANT || (!mStatic)) {
            converter->onCopyBuffer(src, dst);
        }
    }

    return mExecution->onExecute(mWrapInputTensors, outputs);
}

} // namespace MNN
