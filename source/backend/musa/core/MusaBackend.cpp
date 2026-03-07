//
//  MusaBackend.cpp
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "backend/musa/core/MusaBackend.hpp"
#include "MNN_generated.h"

#include <map>
#include <mutex>
#include "core/Macro.h"
#include "shape/SizeComputer.hpp"
#include "core/TensorUtils.hpp"
#include "core/BufferAllocator.hpp"

namespace MNN {
namespace MUSA {

std::map<OpType, MusaBackend::Creator*>* gCreator() {
    static std::map<OpType, MusaBackend::Creator*>* creators = nullptr;
    static std::once_flag gOnce;
    std::call_once(gOnce, [&]() { creators = new std::map<OpType, MusaBackend::Creator*>; });
    return creators;
};

class MusaRuntimeAllocator : public BufferAllocator::Allocator {
public:
    MusaRuntimeAllocator(MusaRuntime* rt) : mRuntime(rt) {}
    virtual ~MusaRuntimeAllocator() = default;
    virtual MemChunk onAlloc(size_t size, size_t align) override {
        return MemChunk(mRuntime->alloc(size), 0);
    }
    virtual void onRelease(MemChunk ptr) override {
        mRuntime->free(ptr.first);
    }
private:
    MusaRuntime* mRuntime;
};

MusaRuntimeWrapper::MusaRuntimeWrapper(BackendConfig::PrecisionMode precision, BackendConfig::PowerMode power, BackendConfig::MemoryMode memory, int deviceId) {
    mMusaRuntime.reset(new MusaRuntime(deviceId));
    if (mMusaRuntime.get()) {
        if (mMusaRuntime->isCreateError()) {
            mIsCreateError = true;
            return;
        }
        std::shared_ptr<BufferAllocator::Allocator> allocator(new MusaRuntimeAllocator(mMusaRuntime.get()));
        mBufferPool.reset(new EagerBufferAllocator(allocator));
    }
    mDefaultPrecision = precision;
    mDefaultMemory = memory;
}

MusaRuntimeWrapper::~MusaRuntimeWrapper() {}

float MusaRuntimeWrapper::onGetMemoryInMB() {
    auto staticMemoryInMB = mBufferPool->totalSize() / 1024.0f / 1024.0f;
    return staticMemoryInMB;
}

std::pair<const void*, size_t> MusaRuntimeWrapper::onGetCache() {
    return mMusaRuntime->makeCache();
}

bool MusaRuntimeWrapper::onSetCache(const void* buffer, size_t size) {
    return mMusaRuntime->setCache(std::make_pair(buffer, size));
}

Backend* MusaRuntimeWrapper::onCreate(const BackendConfig* config, Backend* origin) const {
    auto precision_mode = mDefaultPrecision;
    auto memory_mode = mDefaultMemory;
    if (nullptr != config) {
        precision_mode = config->precision;
        memory_mode = config->memory;
    }
    int precision = 0;
    if (precision_mode == BackendConfig::Precision_Low) {
        precision = 2;
    } else if (precision_mode == BackendConfig::Precision_Normal) {
        precision = 0;
    } else if (precision_mode == BackendConfig::Precision_Low_BF16) {
        precision = 3;
    } else {
        precision = 1;
    }
    return new MusaBackend(mBufferPool, mMusaRuntime, precision, memory_mode);
}

void MusaRuntimeWrapper::onGabageCollect(int level) {
    mBufferPool->release(false);
}

MusaBackend::MusaBackend(std::shared_ptr<BufferAllocator> st,
                         std::shared_ptr<MusaRuntime> rt,
                         int precision, BackendConfig::MemoryMode memory)
    : Backend(MNN_FORWARD_MUSA) {
    mBufferPool.reset(new EagerBufferAllocator(BufferAllocator::Allocator::createRecurse(st.get())));
    mStaticBufferPool = st;
    mMusaRuntime = rt;
    mUseFp16AsFp32 = (precision == 2);
    mPrecision = precision;
    mMemory = memory;
}

MusaBackend::~MusaBackend() {}

MusaRuntime* MusaBackend::getMusaRuntime() {
    MNN_ASSERT(nullptr != mMusaRuntime.get());
    return mMusaRuntime.get();
}

const Runtime* MusaBackend::getRuntime() {
    return (const Runtime*)mMusaRuntime.get();
}

bool MusaBackend::useFp16() const {
    return mUseFp16AsFp32;
}

int MusaBackend::getPrecision() const {
    return mPrecision;
}

BackendConfig::MemoryMode MusaBackend::getMemoryMode() const {
    return mMemory;
}

class MusaMemObj : public Backend::MemObj {
public:
    MusaMemObj(BufferAllocator* allocator, MemChunk points) {
        mPoint = std::move(points);
        mAllocator = allocator;
    }
    virtual ~MusaMemObj() {
        mAllocator->free(mPoint);
    }
    MemChunk chunk() override {
        return mPoint;
    }
private:
    BufferAllocator* mAllocator;
    MemChunk mPoint;
};

int MusaBackend::getBytes(const Tensor* tensor) const {
    auto bytes = tensor->getType().bytes();
    if (mPrecision == 2 || mPrecision == 3) { // Fp16 or Bf16
        if (halide_type_float == tensor->getType().code) {
            bytes = 2;
        }
    }
    auto quant = TensorUtils::getDescribe(tensor)->quantAttr.get();
    if (nullptr != quant && TensorUtils::getDescribe(tensor)->type == DataType_DT_INT8) {
        bytes = 1;
    }
    return bytes;
}

CPUResizeCache* MusaBackend::getCache() {
    return &mCache;
}

Backend::MemObj* MusaBackend::onAcquire(const Tensor* nativeTensor, StorageType storageType) {
    BufferAllocator* allocator = nullptr;
    auto bytes = getBytes(nativeTensor);
    size_t mallocSize = realSize(nativeTensor) * bytes;

    MemChunk buffer;
    if (storageType == DYNAMIC_SEPERATE) {
        buffer = mBufferPool->alloc(mallocSize, true);
        allocator = mBufferPool.get();
    } else if (storageType == DYNAMIC) {
        buffer = mBufferPool->alloc(mallocSize, false);
        allocator = mBufferPool.get();
    } else {
        MNN_ASSERT(storageType == STATIC);
        buffer = mStaticBufferPool->alloc(mallocSize, false);
        allocator = mStaticBufferPool.get();
    }
    if (nullptr == buffer.first) {
        return nullptr;
    }
    auto host = buffer.ptr();
    ((Tensor*)nativeTensor)->buffer().device = (uint64_t)host;
    auto des = TensorUtils::getDescribe(nativeTensor);
    des->extra.offset = buffer.second;
    return new MusaMemObj(allocator, buffer);
}

bool MusaBackend::onClearBuffer() {
    mCache.reset();
    mBufferPool->release(true);
    return true;
}

size_t MusaBackend::realSize(const Tensor* tensor) {
    auto dim = TensorUtils::getDescribe(tensor)->dimensionFormat;
    int pack = 1;
    if (dim == MNN_DATA_FORMAT_NC4HW4) {
        pack = PACK_NUMBER;
        if (getDataType(tensor) == DataType_DT_INT8 || tensor->getType().bytes() == 1) {
            pack = INT8_PACK_NUMBER;
        }
    }
    size_t res = 1;
    for (int i = 0; i < tensor->dimensions(); ++i) {
        size_t l = tensor->length(i);
        if (1 == i) {
            l = UP_DIV(l, pack) * pack;
        }
        res *= l;
    }
    return res;
}

Execution* MusaBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const MNN::Op* op) {
    auto opType = op->type();
    auto creators = gCreator();
    auto iter = creators->find(opType);
    if (iter == creators->end()) {
        if (nullptr != op->name()) {
            MNN_PRINT("MusaBackend Don't support type %s, %s\n", EnumNameOpType(opType), op->name()->c_str());
        } else {
            MNN_PRINT("MusaBackend Don't support type %s\n", EnumNameOpType(opType));
        }
        return NULL;
    }
    auto exe = iter->second->onCreate(inputs, outputs, op, this);
    if (NULL == exe) {
        if (nullptr != op->name()) {
            MNN_PRINT("MusaBackend The Creator Don't support type %s, %s\n", EnumNameOpType(opType), op->name()->c_str());
        } else {
            MNN_PRINT("MusaBackend The Creator Don't support type %s\n", EnumNameOpType(opType));
        }
        return NULL;
    }
    return exe;
}

void MusaBackend::onResizeBegin() {}

ErrorCode MusaBackend::onResizeEnd() {
    return NO_ERROR;
}

void MusaBackend::onExecuteBegin() const {
    mMusaRuntime->activate();
}

void MusaBackend::onExecuteEnd() const {}

void MusaBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    auto& srcBuffer = srcTensor->buffer();
    auto& dstBuffer = dstTensor->buffer();
    
    void* src = (void*)srcBuffer.device;
    void* dst = (void*)dstBuffer.device;
    auto size = realSize(srcTensor) * getBytes(srcTensor);
    
    if (nullptr != src && nullptr != dst) {
        mMusaRuntime->memcpy(dst, src, size, MNNMemcpyDeviceToDevice, true);
    }
}

int MusaBackend::onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) {
    mMusaRuntime->device_sync();
    return 0;
}

DataType MusaBackend::getDataType(const Tensor* tensor) {
    auto dtype = tensor->getType();
    if (dtype.code == halide_type_float && dtype.bits == 32) {
        return DataType_DT_FLOAT;
    } else if (dtype.code == halide_type_float && dtype.bits == 16) {
        return DataType_DT_BFLOAT16;  // Use BF16 as FP16 placeholder
    } else if (dtype.code == halide_type_int && dtype.bits == 8) {
        return DataType_DT_INT8;
    }
    return DataType_DT_FLOAT;
}

bool MusaBackend::addCreator(OpType t, Creator* c) {
    auto creators = gCreator();
    creators->insert(std::make_pair(t, c));
    return true;
}

} // namespace MUSA
} // namespace MNN