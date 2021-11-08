//
//  CUDABackend.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cuda/core/CUDABackend.hpp"
#include "MNN_generated.h"

#include <map>
#include <mutex>
#include "core/Macro.h"
#include "shape/SizeComputer.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace CUDA {

std::map<OpType, CUDABackend::Creator*>* gCreator() {
    static std::map<OpType, CUDABackend::Creator*>* creators = nullptr;
    static std::once_flag gOnce;
    std::call_once(gOnce, [&]() { creators = new std::map<OpType, CUDABackend::Creator*>; });
    return creators;
};
class CUDARuntimeAllocator : public BufferAllocator::Allocator {
public:
    CUDARuntimeAllocator(CUDARuntime* rt) : mRuntime(rt) {
        // Do nothing
    }
    virtual ~ CUDARuntimeAllocator() = default;
    virtual std::pair<void*, int> onAlloc(int size) override {
        return std::make_pair(mRuntime->alloc(size), 0);
    }
    virtual void onRelease(std::pair<void*, int> ptr) override {
        mRuntime->free(ptr.first);
    }
private:
    CUDARuntime* mRuntime;
};
CUDARuntimeWrapper::CUDARuntimeWrapper(BackendConfig::PrecisionMode precision, BackendConfig::PowerMode power) {
    // Shader precision
    if (precision == BackendConfig::Precision_Low) {
        mCUDARuntime.reset(new CUDARuntime(true, -1));
    } else {
        mCUDARuntime.reset(new CUDARuntime(false, -1));
    }
    if (mCUDARuntime.get()) {
        if (mCUDARuntime->isCreateError() == true) {
            mIsCreateError = true;
            return;
        }
        std::shared_ptr<BufferAllocator::Allocator> allocator(new CUDARuntimeAllocator(mCUDARuntime.get()));
        mBufferPool.reset(new BufferAllocator(allocator));
    }
}
CUDARuntimeWrapper::~CUDARuntimeWrapper() {
    // Do nothing
}
float CUDARuntimeWrapper::onGetMemoryInMB() {
    auto staticMemoryInMB = mBufferPool->totalSize() / 1024.0f / 1024.0f;
    return staticMemoryInMB;
}

Backend* CUDARuntimeWrapper::onCreate(const BackendConfig* config) const {
    return new CUDABackend(mBufferPool, mCUDARuntime);
}

void CUDARuntimeWrapper::onGabageCollect(int level) {
    mBufferPool->release(false);
}

CUDABackend::CUDABackend(std::shared_ptr<BufferAllocator> st,
                         std::shared_ptr<CUDARuntime> rt)
    : Backend(MNN_FORWARD_CUDA) {
    mBufferPool.reset(new BufferAllocator(BufferAllocator::Allocator::createRecurse(st.get())));
    mStaticBufferPool = st;
    mCUDARuntime      = rt;
}

CUDABackend::~CUDABackend() {
#ifdef LOG_VERBOSE
    MNN_PRINT("enter CUDABackend::~CUDABackend \n");
#endif
}

CUDARuntime* CUDABackend::getCUDARuntime() {
    MNN_ASSERT(nullptr != mCUDARuntime.get());
    return mCUDARuntime.get();
}

bool CUDABackend::onAcquireBuffer(const Tensor* nativeTensor, StorageType storageType) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start CUDABackend::onAcquireBuffer !\n");
#endif
    int mallocSize = realSize(nativeTensor) * nativeTensor->getType().bytes();
    std::pair<void*, int> buffer;
    if (storageType == DYNAMIC_SEPERATE) {
        buffer                              = mBufferPool->alloc(mallocSize, true);
    } else if (storageType == DYNAMIC) {
        buffer                              = mBufferPool->alloc(mallocSize, false);
    } else {
        MNN_ASSERT(storageType == STATIC);
        buffer                              = mStaticBufferPool->alloc(mallocSize, false);
    }
    if(nullptr == buffer.first) {
        return false;
    };
    auto host = (uint8_t*)buffer.first + buffer.second;
    ((Tensor*)nativeTensor)->buffer().device = (uint64_t)host;
    auto des = TensorUtils::getDescribe(nativeTensor);
    des->extra.offset = buffer.second;
    return true;
}

bool CUDABackend::onReleaseBuffer(const Tensor* nativeTensor, StorageType storageType) {
    if (storageType == DYNAMIC_SEPERATE) {
        return true;
    }
    auto buffer = (uint8_t*)nativeTensor->deviceId();
    auto des = TensorUtils::getDescribe(nativeTensor);
    auto pointer = std::make_pair(buffer - des->extra.offset, des->extra.offset);

    if (storageType == DYNAMIC) {
        mBufferPool->free(pointer);
        return true;
    }
    if (storageType == STATIC) {
        mStaticBufferPool->free(pointer);
    }
    return true;
}

bool CUDABackend::onClearBuffer() {
    mBufferPool->release(true);
    return true;
}
size_t CUDABackend::realSize(const Tensor* tensor) {
    size_t res = 1;
    for (int i = 0; i < tensor->dimensions(); ++i) {
        res *= tensor->length(i);
    }
    return res;
}

std::pair<float, bool> CUDABackend::onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                              const MNN::Op* op) {
    auto creators = gCreator();
    auto iter     = creators->find(op->type());
    if (iter == creators->end()) {
        return std::make_pair(0.0f, false);
    }
    const float defaultScheduleTime = 0.05f;
    // FIXME: Compute in future
    auto flops = 0.0f;
    auto computeFlops = mCUDARuntime->flops();
    return std::make_pair(defaultScheduleTime + flops / 1024.0f / computeFlops * 1000.0f, true);
}
Execution* CUDABackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const MNN::Op* op) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start CUDABackend::onCreate \n");
#endif
    auto creators = gCreator();
    auto iter     = creators->find(op->type());

    if (iter == creators->end()) {
        if (nullptr != op->name()) {
            MNN_PRINT("Don't support type %s, %s\n", EnumNameOpType(op->type()), op->name()->c_str());
        } else {
            MNN_PRINT("Don't support type %s\n", EnumNameOpType(op->type()));
        }
        return NULL;
    }

    auto exe = iter->second->onCreate(inputs, outputs, op, this);
    if (NULL == exe) {
        if (nullptr != op->name()) {
            MNN_PRINT("The Creator Don't support type %s, %s\n", EnumNameOpType(op->type()), op->name()->c_str());
        } else {
            MNN_PRINT("The Creator Don't support type %s\n", EnumNameOpType(op->type()));
        }
        return NULL;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("End CUDABackend::onCreate \n");
#endif
    return exe;
}

void CUDABackend::onResizeBegin() {
}

void CUDABackend::onResizeEnd() {
}

void CUDABackend::onExecuteBegin() const {
}

void CUDABackend::onExecuteEnd() const {
}

void CUDABackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    auto srcDimensionFormat = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
    auto srcDevice          = srcTensor->deviceId() != 0;

    auto dstDimensionFormat = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
    auto dstDevice          = dstTensor->deviceId() != 0;
    if (srcDevice && srcDimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        srcDimensionFormat = MNN_DATA_FORMAT_NCHW;
    }
    if (dstDevice && dstDimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        dstDimensionFormat = MNN_DATA_FORMAT_NCHW;
    }
    auto needSize = realSize(srcTensor) * srcTensor->getType().bytes();
    std::shared_ptr<Tensor> srcTempTensor;
    std::shared_ptr<Tensor> dstTempTensor;
    
    if (srcTensor->deviceId() != 0 && dstTensor->deviceId() != 0) {
        mCUDARuntime->memcpy((void*)(dstTensor->deviceId()), (void*)(srcTensor->deviceId()), needSize,
                            MNNMemcpyDeviceToDevice, true);
    }
    if (srcTensor->deviceId() != 0 && dstTensor->deviceId() == 0) {
        if(srcDimensionFormat != dstDimensionFormat) {

            dstTempTensor.reset(new Tensor(srcTensor, srcTensor->getDimensionType(), true));
            mCUDARuntime->memcpy(dstTempTensor->host<void>(), (void*)(srcTensor->deviceId()), needSize, MNNMemcpyDeviceToHost,
                             true);
            MNNCPUCopyBuffer(dstTempTensor.get(), dstTensor);
        } else {
            mCUDARuntime->memcpy(dstTensor->host<void>(), (void*)(srcTensor->deviceId()), needSize, MNNMemcpyDeviceToHost,
                             true);
        }
    }
    if (srcTensor->deviceId() == 0 && dstTensor->deviceId() != 0) {
        if (srcDimensionFormat != dstDimensionFormat) {
            srcTempTensor.reset(new Tensor(dstTensor, dstTensor->getDimensionType(), true));
            MNNCPUCopyBuffer(srcTensor, srcTempTensor.get());
            srcTensor = srcTempTensor.get();
        }
        mCUDARuntime->memcpy((void*)(dstTensor->deviceId()), srcTensor->host<void>(), needSize, MNNMemcpyHostToDevice,
                             true);
    }
    return;
}

bool CUDABackend::addCreator(OpType t, Creator* c) {
    auto map = gCreator();
    if (map->find(t) != map->end()) {
        MNN_PRINT("Error: %d type has be added\n", t);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}

} // namespace CUDA
} // namespace MNN
