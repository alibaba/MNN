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

WrapExecution::WrapExecution(Backend* CPUBackend, std::shared_ptr<Execution> execution)
    : Execution(execution->backend()), mCPUBackend(CPUBackend), mExecution(execution) {
    mValid = execution->valid();
}

ErrorCode WrapExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mWrapInputTensors.clear();
    mInputMaps.clear();

    auto dstBackend = mExecution->backend();
    for (int i = 0; i < inputs.size(); ++i) {
        auto inputTensor = inputs[i];
        auto srcBackend  = TensorUtils::getDescribe(inputTensor)->backend;

        // CPU -> CPU or XPU -> XPU
        if (srcBackend == dstBackend) {
            mWrapInputTensors.emplace_back(inputTensor);
        }
        // CPU -> XPU
        else if (srcBackend == mCPUBackend) {
            std::shared_ptr<Tensor> wrapTensor(new Tensor);
            TensorUtils::copyShape(inputTensor, wrapTensor.get(), true);
            wrapTensor->buffer().type = inputTensor->buffer().type;
            mInputMaps.emplace_back(std::make_tuple(dstBackend, dstBackend, inputTensor, wrapTensor));
            mWrapInputTensors.emplace_back(wrapTensor.get());
        }
        // XPU -> CPU
        else if (dstBackend == mCPUBackend) {
            std::shared_ptr<Tensor> wrapTensor(new Tensor);
            TensorUtils::copyShape(inputTensor, wrapTensor.get(), true);
            wrapTensor->buffer().type = inputTensor->buffer().type;
            mInputMaps.emplace_back(std::make_tuple(mCPUBackend, srcBackend, inputTensor, wrapTensor));
            mWrapInputTensors.emplace_back(wrapTensor.get());
        }
        // XPU -> CPU -> XPU'
        else {
            std::shared_ptr<Tensor> midTensor(new Tensor);
            std::shared_ptr<Tensor> wrapTensor(new Tensor);
            TensorUtils::copyShape(inputTensor, midTensor.get(), true);
            TensorUtils::copyShape(inputTensor, wrapTensor.get(), true);
            TensorUtils::getDescribe(midTensor.get())->usage = TensorUtils::getDescribe(inputTensor)->usage;
            midTensor->buffer().type                           = inputTensor->buffer().type;
            wrapTensor->buffer().type                          = inputTensor->buffer().type;
            mInputMaps.emplace_back(std::make_tuple(mCPUBackend, srcBackend, inputTensor, midTensor));
            mInputMaps.emplace_back(std::make_tuple(dstBackend, dstBackend, midTensor.get(), wrapTensor));
            mWrapInputTensors.emplace_back(wrapTensor.get());
        }
    }

    for (int i = 0; i < outputs.size(); ++i) {
        MNN_ASSERT(TensorUtils::getDescribe(outputs[i])->backend == dstBackend);
    }
    bool memoryAllocSuccess = true;
    // acquire memory, copy const tensors
    for (auto& iter : mInputMaps) {
        auto backend   = std::get<0>(iter);
        auto converter = std::get<1>(iter);
        auto src       = std::get<2>(iter);
        auto dst       = std::get<3>(iter).get();

        if (TensorUtils::getDescribe(src)->usage == TensorUsage::CONST) {
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
    auto result = mExecution->onResize(mWrapInputTensors, outputs);

    // release memory
    for (auto& iter : mInputMaps) {
        auto backend = std::get<0>(iter);
        auto dst     = std::get<3>(iter).get();

        if (TensorUtils::getDescribe(dst)->usage == TensorUsage::CONST) {
            backend->onReleaseBuffer(dst, Backend::DYNAMIC_SEPERATE);
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
        auto converter = std::get<1>(iter);
        auto src       = std::get<2>(iter);
        auto dst       = std::get<3>(iter).get();
        if (TensorUtils::getDescribe(src)->usage != TensorUsage::CONST) {
            converter->onCopyBuffer(src, dst);
        }
    }
    mExecution->onExecute(mWrapInputTensors, outputs);
    return NO_ERROR;
}

} // namespace MNN
