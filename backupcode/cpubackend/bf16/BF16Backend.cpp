//
//  BF16Backend.cpp
//  MNN
//
//  Created by MNN on 2020/01/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>

#include "BF16Functions.hpp"
#include "BF16Backend.hpp"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {

// The Function Will be Called in init
void registerBF16Backend() {
    BF16Functions::init();
}

BF16Backend::BF16Backend(const CPURuntime* runtime) : CPUBackend(runtime, BackendConfig::Precision_Low, BackendConfig::Memory_Normal, MNN_FORWARD_CPU_EXTENSION) {
    mCoreFunctions = BF16Functions::get();
}

BF16Backend::~BF16Backend() {
    // nothing to do
}

Execution* BF16Backend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const MNN::Op* op) {
    for (auto t : outputs) {
        if (t->getType().code != halide_type_float) {
            return nullptr;
        }
    }
    if (outputs.size() == 1) {
        if (TensorUtils::getDescribe(outputs[0])->quantAttr != nullptr) {
            return nullptr;
        }
    }
    bool originCreate = OpCommonUtils::opCompabilityForLowp(op, 2);
    if (originCreate) {
        return CPUBackend::onCreate(inputs, outputs, op);
    }
    return nullptr;
}

static size_t _getAliginSize(const halide_buffer_t& buffer, MNN_DATA_FORMAT format) {
    // The default data type of input tensor for arm82 backend is FLOAT32.
    // However, BF16Backend default data type is FLOAT16, so check whether data type is FLOAT32,
    // then divide size by 2
    size_t size          = sizeof(int16_t);
    const int dimensions = buffer.dimensions;
    for (int i = 0; i < dimensions; i++) {
        int currentDimSize = buffer.dim[i].extent;
        if (format == MNN_DATA_FORMAT_NC4HW4 && 1 == i) {
            currentDimSize = ALIGN_UP4(currentDimSize);
        }
        size *= currentDimSize;
    }
    return size;
}

Backend::MemObj* BF16Backend::onAcquire(const Tensor* nativeTensor, StorageType storageType) {
    // arm82 backend tensor data type is fp16 default
    auto tensor = const_cast<Tensor*>(nativeTensor);
    auto& buffer = tensor->buffer();
    if (buffer.type != halide_type_of<float>()) {
        return CPUBackend::onAcquire(nativeTensor, storageType);
    }
    auto res = allocBuffer(_getAliginSize(buffer, TensorUtils::getDescribe(nativeTensor)->dimensionFormat), (Tensor*)nativeTensor, storageType);
    if (!res) {
        return nullptr;
    }
    // Set mask in device for easy to determine
    buffer.device = 1;
    return res;
}

void BF16Backend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    auto& ib     = srcTensor->buffer();
    auto& ob     = dstTensor->buffer();
    if (ib.type.code != halide_type_float) {
        CPUBackend::onCopyBuffer(srcTensor, dstTensor);
        return;
    }
    auto source = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
    auto dest   = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
    auto srcType = MNN_FORWARD_CPU;
    if (ib.device != 0) {
        srcType = MNN_FORWARD_CPU_EXTENSION;
    }
    auto dstType = MNN_FORWARD_CPU;
    if (ob.device != 0) {
        dstType = MNN_FORWARD_CPU_EXTENSION;
    }
    if (srcType == dstType) {
        ErrorCode code = ErrorCode::NO_ERROR;
        auto tup = CPUTensorConverter::splitDimensions(srcTensor->buffer(), source);
        int area = std::get<1>(tup), batch = std::get<0>(tup), channel = std::get<2>(tup);
        if (srcType == MNN_FORWARD_CPU) {
            code = CPUTensorConverter::convert(srcTensor->host<void>(), dstTensor->host<void>(), source, dest, batch, area, channel, 4, MNNGetCoreFunctions());
        } else {
            code = CPUTensorConverter::convert(srcTensor->host<void>(), dstTensor->host<void>(), source, dest, batch, area, channel, 2, mCoreFunctions);
        }
        MNN_ASSERT(code == ErrorCode::NO_ERROR);
        return;
    }
    // Use CPU Copy to turn save format
    std::shared_ptr<Tensor> tempTensor;
    if (source != dest) {
        if (srcType == MNN_FORWARD_CPU) {
            tempTensor.reset(Tensor::create<float>(dstTensor->shape(), nullptr, TensorUtils::getDimType(dstTensor)));
            MNNCPUCopyBuffer(srcTensor, tempTensor.get());
            srcTensor = tempTensor.get();
            source = dest;
        } else {
            tempTensor.reset(Tensor::create<float>(srcTensor->shape(), nullptr, TensorUtils::getDimType(srcTensor)), [dstTensor](void* ptr) {
                auto tempT = (Tensor*)ptr;
                MNNCPUCopyBuffer(tempT, dstTensor);
                delete tempT;
            });
            dstTensor = tempTensor.get();
            dest = source;
        }
    }
    //MNN_PRINT("%d, %d - %d, %d\n", source, srcType, dest, dstType);
    // The format is the same, just convert fp32-fp16
    const int elemenSize = srcTensor->elementSize();
    // copy and quantize/dequantize data
    if (srcType == MNN_FORWARD_CPU) {
        const auto src = srcTensor->host<float>();
        auto dst       = dstTensor->host<int16_t>();
        BF16Functions::get()->MNNFp32ToLowp(src, dst, elemenSize);
        return;
    }
    if (srcType == MNN_FORWARD_CPU_EXTENSION) {
        const auto src = srcTensor->host<int16_t>();
        auto dst       = dstTensor->host<float>();
        BF16Functions::get()->MNNLowpToFp32(src, dst, elemenSize);
        return;
    }
    return;
}

} // namespace MNN
