//
//  MusaBackend.cpp
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "MusaBackend.hpp"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
#include <string.h>
#include <map>

namespace MNN {
namespace MUSA {

static std::map<OpType, MusaBackend::Creator*>* gCreator = nullptr;

MusaBackend::MusaBackend(std::shared_ptr<BufferAllocator> st, std::shared_ptr<MusaRuntime> rt, int precisionLevel, BackendConfig::MemoryMode memoryLevel)
    : Backend(MNN_FORWARD_MUSA), mBufferPool(st), mStaticBufferPool(std::make_shared<StaticBufferAllocator>(st.get())), mMusaRuntime(rt), mPrecision(precisionLevel), mMemory(memoryLevel) {
}

MusaBackend::~MusaBackend() {
    // Destructor
}

MusaRuntime* MusaBackend::getMusaRuntime() {
    return mMusaRuntime.get();
}

const Runtime* MusaBackend::getRuntime() {
    return mMusaRuntime.get();
}

Backend::MemObj* MusaBackend::onAcquire(const Tensor* nativeTensor, StorageType storageType) {
    auto dimType = TensorUtils::getDescribe(nativeTensor)->dimensionFormat;
    auto& buffer = nativeTensor->buffer();
    size_t size  = 0;
    if (storageType == Storage_Internal) {
        size = mMusaRuntime->getMemoryUsage(nativeTensor);
    } else {
        size = nativeTensor->size();
    }

    if (size <= 0) {
        return nullptr;
    }

    MemObj* result = new MemObj;
    result->storage = storageType;
    result->size = size;
    
    void* ptr = nullptr;
    if (storageType == Storage_Internal) {
        ptr = mBufferPool->alloc(size);
    } else {
        ptr = mMusaRuntime->alloc(size);
    }
    
    if (nullptr == ptr) {
        delete result;
        return nullptr;
    }
    
    result->base = (uint8_t*)ptr;
    TensorUtils::getDescribe(nativeTensor)->memory = result;
    return result;
}

bool MusaBackend::onClearBuffer() {
    mBufferPool->clear();
    mStaticBufferPool->clear();
    return true;
}

Execution* MusaBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {
    auto type = op->type();
    auto iter = gCreator->find(type);
    if (iter == gCreator->end()) {
        return nullptr;
    }
    return iter->second->onCreate(inputs, outputs, op, this);
}

void MusaBackend::onResizeBegin() {
    mBufferPool->onResizeBegin();
}

ErrorCode MusaBackend::onResizeEnd() {
    mBufferPool->onResizeEnd();
    return NO_ERROR;
}

void MusaBackend::onExecuteBegin() const {
    mMusaRuntime->activate();
}

void MusaBackend::onExecuteEnd() const {
    // Device sync if needed
}

void MusaBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    auto srcType = TensorUtils::getDescribe(srcTensor)->memory->storage;
    auto dstType = TensorUtils::getDescribe(dstTensor)->memory->storage;
    
    void* src = TensorUtils::getDescribe(srcTensor)->memory->base;
    void* dst = TensorUtils::getDescribe(dstTensor)->memory->base;
    size_t size = srcTensor->size();
    
    if (srcType == Storage_Internal && dstType == Storage_Internal) {
        mMusaRuntime->memcpy(dst, src, size, MNNMemcpyDeviceToDevice, true);
    } else if (srcType == Storage_Internal && dstType == Storage_External) {
        mMusaRuntime->memcpy(dst, src, size, MNNMemcpyDeviceToHost, true);
    } else if (srcType == Storage_External && dstType == Storage_Internal) {
        mMusaRuntime->memcpy(dst, src, size, MNNMemcpyHostToDevice, true);
    } else {
        ::memcpy(dst, src, size);
    }
}

int MusaBackend::onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) {
    // Sync implementation
    return 0;
}

size_t MusaBackend::realSize(const Tensor* tensor) {
    return TensorUtils::getDescribe(tensor)->elements;
}

int MusaBackend::getBytes(const Tensor* tensor) const {
    return TensorUtils::getDescribe(tensor)->type.bytes();
}

CPUResizeCache* MusaBackend::getCache() {
    return &mCache;
}

bool MusaBackend::useFp16() const {
    return mPrecision == BackendConfig::Precision_High;
}

int MusaBackend::getPrecision() const {
    return mPrecision;
}

BackendConfig::MemoryMode MusaBackend::getMemoryMode() const {
    return mMemory;
}

DataType MusaBackend::getDataType(const Tensor* tensor) {
    auto dtype = tensor->getType();
    if (dtype.bits == 32) {
        return DataType_FLOAT32;
    } else if (dtype.bits == 16) {
        return DataType_FLOAT16;
    } else if (dtype.code == halide_type_int && dtype.bits == 8) {
        return DataType_INT8;
    }
    return DataType_FLOAT32;
}

bool MusaBackend::addCreator(OpType t, Creator* c) {
    if (nullptr == gCreator) {
        gCreator = new std::map<OpType, Creator*>;
    }
    gCreator->insert(std::make_pair(t, c));
    return true;
}

} // namespace MUSA
} // namespace MNN
