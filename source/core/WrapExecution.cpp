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

Tensor* WrapExecution::_getCopyTensor(Tensor* inputTensor) {
    auto dstBackend = mExecution->backend();
    auto inputDes   = TensorUtils::getDescribe(inputTensor);
    auto srcBackend = inputDes->backend;
    if (nullptr == srcBackend) {
        srcBackend = mCPUBackend;
    }
    // CPU -> CPU or XPU -> XPU
    //if (srcBackend == dstBackend) {
    if (srcBackend->type() == dstBackend->type()) {
        return inputTensor;
    }
    auto iter = mInputMaps.find(inputTensor);
    if (iter != mInputMaps.end()) {
        return std::get<2>(iter->second).get();
    }
    // CPU -> XPU
    if (srcBackend->type() == mCPUBackend->type()) {
        std::shared_ptr<Tensor> wrapTensor(new Tensor);
        TensorUtils::copyShape(inputTensor, wrapTensor.get(), true);
        TensorUtils::adjustTensorForCompability(wrapTensor.get());
        wrapTensor->buffer().type = inputTensor->buffer().type;
        TensorUtils::getDescribe(wrapTensor.get())->quantAttr = TensorUtils::getDescribe(inputTensor)->quantAttr;
        mInputMaps.insert(std::make_pair(inputTensor, std::make_tuple(dstBackend, dstBackend, wrapTensor)));
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
    for (int i = 0; i < inputs.size(); ++i) {
        auto inputTensor = inputs[i];
        auto des         = TensorUtils::getDescribe(inputTensor);
        if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
            MNN_ASSERT(inputs.size() == 1);
            mWrapForRaster.reset(new Tensor);
            TensorUtils::copyShape(inputTensor, mWrapForRaster.get(), true);
            mWrapForRaster->buffer().type = inputTensor->buffer().type;
            auto wrapDes                  = TensorUtils::getDescribe(mWrapForRaster.get());
            wrapDes->memoryType           = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            wrapDes->regions              = des->regions;
            for (auto& r : wrapDes->regions) {
                r.origin = _getCopyTensor(r.origin);
            }
            mWrapInputTensors[i] = mWrapForRaster.get();
        } else {
            mWrapInputTensors[i] = _getCopyTensor(inputTensor);
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
    auto result = mExecution->onResize(mWrapInputTensors, outputs);

    // release memory
    for (auto& iter : mInputMaps) {
        auto backend = std::get<0>(iter.second);
        auto dst     = std::get<2>(iter.second).get();

        if (TensorUtils::getDescribe(dst)->usage == TensorUsage::CONSTANT && mStatic) {
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
                                     const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, halide_type_t runT)
                                   : Execution(backend), runType(runT), mCreator(creator), mType(op->type()), mInputs(inputs) {
    std::vector<int> types(inputs.size());
    for (int i = 0; i < inputs.size(); i++) {
        types[i] = TensorUtils::HaildeTypeToDataType(inputs[i]->getType());
        inputs[i]->setType(TensorUtils::HaildeTypeToDataType(runType));
    }
    mExecution.reset(mCreator->onCreate(inputs, outputs, op, backend));
    for (int i = 0; i < inputs.size(); i++) {
        inputs[i]->setType(types[i]);
    }
}
ErrorCode CastWrapExecution::onResize(const std::vector<Tensor*>& inputs,
                                      const std::vector<Tensor*>& outputs) {
    for (auto output : outputs) {
        output->setType(TensorUtils::HaildeTypeToDataType(runType));
    }
    mWrapInputs.clear();
    mCasts.clear();
    mScales.clear();
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
        if (input->getType() == runType || !OpCommonUtils::opNeedContent(mType, i) || input->getType() == halide_type_of<int>()) {
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
        TensorUtils::getDescribe(wrapTensor.get())->quantAttr = TensorUtils::getDescribe(input)->quantAttr;
        wrapTensor->buffer().type = runType;
        bool memoryAllocSuccess = backend()->onAcquireBuffer(wrapTensor.get(), Backend::DYNAMIC);
        if (!memoryAllocSuccess) {
            return {};
        }
        mWrapInputs.push_back(wrapTensor.get());
        auto wrapPointer = wrapTensor.get();
        mCasts.insert(std::make_pair(input, wrapTensor.get()));
        cachedCastTensor.insert(std::make_pair(input, wrapTensor.get()));
        mWrapInputTensor.emplace_back(std::move(wrapTensor));
        auto& quantAttr = TensorUtils::getDescribe(input)->quantAttr;
        float scale = runType == halide_type_of<float>() ? quantAttr->scale : 1/quantAttr->scale;
        // set 4xscale for SSE compute
        mScales[input] = std::vector<float>(4, scale);
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
    for (auto& iter : mCasts) {
        if (TensorUtils::getDescribe(iter.first)->useCount <= 1) {
            backend()->onReleaseBuffer(iter.second, Backend::DYNAMIC);
        }
    }
    return res;
}
ErrorCode CastWrapExecution::onExecute(const std::vector<Tensor*>& inputs,
                                       const std::vector<Tensor*>& outputs) {
    for (const auto& iter : mCasts) {
        auto input = iter.first;
        auto output = iter.second;
        auto& quantAttr = TensorUtils::getDescribe(input)->quantAttr;
        MNN_ASSERT(quantAttr != nullptr);
        auto cpuBackend = ((CPUBackend*)backend());
        int size = cpuBackend->getTensorSize(input);
        auto numberThread = cpuBackend->threadNumber();
        if (numberThread == 1) {
            CPUCastCreator::cast(input, output, size);
            continue;
        }
        int sizeQuad = size / 4;
        int sizeDivide = sizeQuad / numberThread;
        int remain = sizeDivide * numberThread * 4;
        auto scale = mScales[input].data();
        if (runType == halide_type_of<float>()) {
            const auto inputDataPtr = input->host<int8_t>();
            auto outputDataPtr      = output->host<float>();
            if (sizeDivide > 0) {
                MNN_CONCURRENCY_BEGIN(tId, numberThread) {
                    const auto srcChannelPtr   = inputDataPtr + tId * sizeDivide * 4;
                    auto dstChannlePtr         = outputDataPtr + tId * sizeDivide * 4;
                    MNNInt8ScaleToFloat(dstChannlePtr, srcChannelPtr, scale, sizeDivide, quantAttr->zero);
                }
                MNN_CONCURRENCY_END();
            }
            for (int i = remain; i < size; i++) {
                outputDataPtr[i] = (inputDataPtr[i] - quantAttr->zero) * scale[0];
            }
        } else {
            const auto inputDataPtr = input->host<float>();
            auto outputDataPtr      = output->host<int8_t>();
            if (sizeDivide > 0) {
                MNN_CONCURRENCY_BEGIN(tId, numberThread) {
                    const auto srcChannelPtr   = inputDataPtr + tId * sizeDivide * 4;
                    auto dstChannlePtr         = outputDataPtr + tId * sizeDivide * 4;
                    MNNFloat2Int8(srcChannelPtr, dstChannlePtr, sizeDivide, scale, quantAttr->min, quantAttr->max, quantAttr->zero);
                }
                MNN_CONCURRENCY_END();
            }
            int number = (size - remain) / 4;
            MNNFloat2Int8(inputDataPtr + remain, outputDataPtr + remain, number, scale, quantAttr->min, quantAttr->max, quantAttr->zero);
            remain = number * 4 + remain;
            if (remain < size) {
                float srcTmp[4];
                int8_t dstTmp[4];
                for (int i = remain; i < size; i++) {
                    srcTmp[i - remain] = inputDataPtr[i];
                }
                MNNFloat2Int8(srcTmp, dstTmp, 1, scale, quantAttr->min, quantAttr->max, quantAttr->zero);
                for (int i = remain; i < size; i++) {
                    outputDataPtr[i] = dstTmp[i - remain];
                }
            }
        }
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
    *dst = new CastWrapExecution(bn, runType, op, exe);
    return true;
}

CheckNANExecution::CheckNANExecution(Execution* exe) : Execution(exe->backend()) {
    mExecution = exe;
    mValid = exe->valid();
}

CheckNANExecution::~CheckNANExecution() {
    delete mExecution;
}

ErrorCode CheckNANExecution::onResize(const std::vector<Tensor*>& inputs,
                                      const std::vector<Tensor*>& outputs) {
    return mExecution->onResize(inputs, outputs);
}

ErrorCode CheckNANExecution::onExecute(const std::vector<Tensor*>& inputs,
                                       const std::vector<Tensor*>& outputs) {
    for (auto tensor : inputs) {
        if (halide_type_float != tensor->getType().code) {
            continue;
        }
        if (TensorUtils::getDescribe(tensor)->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
            continue;
        }
#define MNN_IS_INF(x) (fabs(x) == INFINITY)
#define MNN_IS_NAN(x) ((x) != (x))
        auto size = tensor->elementSize();
        auto ptr  = tensor->host<float>();
        for (int i = 0; i < size; ++i) {
            auto value = ptr[i];
            if (MNN_IS_INF(value) || MNN_IS_NAN(value)) {
                return INVALID_VALUE;
            }
        }
    }
    auto code = mExecution->onExecute(inputs, outputs);
    if (NO_ERROR != code) {
        return code;
    }
    for (auto tensor : outputs) {
        if (halide_type_float != tensor->getType().code) {
            continue;
        }
        auto size = tensor->elementSize();
        auto ptr  = tensor->host<float>();
        for (int i = 0; i < size; ++i) {
            auto value = ptr[i];
            if (MNN_IS_INF(value) || MNN_IS_NAN(value)) {
                return INVALID_VALUE;
            }
        }
    }
    return NO_ERROR;
}

} // namespace MNN
