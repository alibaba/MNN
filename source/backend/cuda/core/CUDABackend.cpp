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

static std::once_flag gOnce;
std::map<OpType, CUDABackend::Creator*>* gCreator() {
    static std::map<OpType, CUDABackend::Creator*>* creators = nullptr;
    std::call_once(gOnce, [&]() { creators = new std::map<OpType, CUDABackend::Creator*>; });
    return creators;
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
        mBufferPool.reset(new BufferPool(mCUDARuntime.get()));
        mStaticBufferPool.reset(new BufferPool(mCUDARuntime.get()));
    }
}
CUDARuntimeWrapper::~CUDARuntimeWrapper() {
    // Do nothing
}
Backend* CUDARuntimeWrapper::onCreate() const {
    return new CUDABackend(mBufferPool, mStaticBufferPool, mCUDARuntime);
}

void CUDARuntimeWrapper::onGabageCollect(int level) {
    mStaticBufferPool->release(false);
    if (level > 50) {
        mBufferPool->release(false);
    }
}

CUDABackend::CUDABackend(std::shared_ptr<BufferPool> dy, std::shared_ptr<BufferPool> st,
                         std::shared_ptr<CUDARuntime> rt)
    : Backend(MNN_FORWARD_CUDA) {
    mBufferPool       = dy;
    mStaticBufferPool = st;
    mCUDARuntime      = rt;
}

CUDABackend::~CUDABackend() {
#ifdef LOG_VERBOSE
    MNN_PRINT("enter CUDABackend::~CUDABackend \n");
#endif
    for (auto p : mStatic) {
        mStaticBufferPool->free(p);
    }
    for (auto p : mDynamic) {
        mBufferPool->free(p);
    }
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
    if (storageType == DYNAMIC_SEPERATE) {
        auto buffer                              = mBufferPool->alloc(mallocSize, true);
        ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer;
    } else if (storageType == DYNAMIC) {
        auto buffer                              = mBufferPool->alloc(mallocSize, false);
        ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer;
    } else {
        MNN_ASSERT(storageType == STATIC);
        auto buffer                              = mStaticBufferPool->alloc(mallocSize, false);
        ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer;
    }
    MNN_ASSERT(0 != ((Tensor*)nativeTensor)->buffer().device);
    if (STATIC == storageType) {
        mStatic.insert((void*)nativeTensor->buffer().device);
    } else {
        mDynamic.insert((void*)nativeTensor->buffer().device);
    }
    return true;
}

bool CUDABackend::onReleaseBuffer(const Tensor* nativeTensor, StorageType storageType) {
    if (storageType == DYNAMIC_SEPERATE) {
        return true;
    }
    auto buffer = nativeTensor->deviceId();
    if (storageType == DYNAMIC) {
        mDynamic.erase((void*)buffer);
        mBufferPool->free((void*)buffer);
        return true;
    }
    if (storageType == STATIC) {
        mStatic.erase((void*)buffer);
        mStaticBufferPool->free((void*)buffer);
    }
    return true;
}

bool CUDABackend::onClearBuffer() {
    for (auto p : mDynamic) {
        mBufferPool->free(p);
    }
    mDynamic.clear();
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
#ifndef MNN_BUILD_MINI
    auto flops                      = SizeComputer::computeFlops(op, inputs, outputs);
#else
    auto flops = 0.0f;
#endif
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
            MNN_PRINT("The Creator Don't support type %d, %s\n", op->type(), op->name()->c_str());
        } else {
            //            MNN_PRINT("The Creator Don't support type %s\n", EnumNameOpType(op->type()));
        }
        return NULL;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("End CUDABackend::onCreate \n");
#endif
    return exe;
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
    if (srcDimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        srcTempTensor.reset(new Tensor(srcTensor, Tensor::CAFFE, true));
        MNNCPUCopyBuffer(srcTensor, srcTempTensor.get());
        srcTensor = srcTempTensor.get();
    }
    if (dstDimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        dstTempTensor.reset(new Tensor(dstTensor, Tensor::CAFFE, true), [dstTensor](void* ptr) {
            auto src = (Tensor*)ptr;
            MNNCPUCopyBuffer(src, dstTensor);
            delete src;
        });
        dstTensor = dstTempTensor.get();
    }
    if (srcTensor->deviceId() != 0 && dstTensor->deviceId() != 0) {
        mCUDARuntime->memcpy((void*)(dstTensor->deviceId()), (void*)(srcTensor->deviceId()), needSize,
                             MNNMemcpyDeviceToDevice, true);
    }
    if (srcTensor->deviceId() != 0 && dstTensor->deviceId() == 0) {
        mCUDARuntime->memcpy(dstTensor->host<void>(), (void*)(srcTensor->deviceId()), needSize, MNNMemcpyDeviceToHost,
                             true);
    }
    if (srcTensor->deviceId() == 0 && dstTensor->deviceId() != 0) {
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
