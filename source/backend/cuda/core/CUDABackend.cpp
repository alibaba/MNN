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
#include "execution/Raster.cuh"
#include "execution/Transpose.cuh"
#include "execution/MNNCUDADefine.hpp"

#include "CUDATools.hpp"

// #define MNN_CUDA_COPY_DEBUG

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
    virtual std::pair<void*, size_t> onAlloc(size_t size, size_t align) override {
        return std::make_pair(mRuntime->alloc(size), 0);
    }
    virtual void onRelease(std::pair<void*, size_t> ptr) override {
        mRuntime->free(ptr.first);
    }
private:
    CUDARuntime* mRuntime;
};
CUDARuntimeWrapper::CUDARuntimeWrapper(BackendConfig::PrecisionMode precision, BackendConfig::PowerMode power) {
    // TODO: Search CUDA Device info and use best one
    mCUDARuntime.reset(new CUDARuntime(-1));
#ifdef LOG_VERBOSE
    MNN_PRINT("create cuda runtime:%p\n", mCUDARuntime.get());
#endif
    if (mCUDARuntime.get()) {
        if (mCUDARuntime->isCreateError() == true) {
            mIsCreateError = true;
            return;
        }
        std::shared_ptr<BufferAllocator::Allocator> allocator(new CUDARuntimeAllocator(mCUDARuntime.get()));
        mBufferPool.reset(new BufferAllocator(allocator));
    }
    mDefaultPrecision = precision;
}
CUDARuntimeWrapper::~CUDARuntimeWrapper() {
    // Do nothing
}
float CUDARuntimeWrapper::onGetMemoryInMB() {
    auto staticMemoryInMB = mBufferPool->totalSize() / 1024.0f / 1024.0f;
    return staticMemoryInMB;
}

Backend* CUDARuntimeWrapper::onCreate(const BackendConfig* config) const {
#ifdef LOG_VERBOSE
    MNN_PRINT("cudaruntime:%p, create CUDABackend\n", this);
#endif
    auto mode = mDefaultPrecision;
    if (nullptr != config) {
        mode = config->precision;
    }
    int precision = 0; 
    if(mode == BackendConfig::Precision_Low) {
        precision = 2;
    } else if(mode == BackendConfig::Precision_Normal) {
        precision = 0;
    } else {
        precision = 1;
    }

    return new CUDABackend(mBufferPool, mCUDARuntime, precision);
}

void CUDARuntimeWrapper::onGabageCollect(int level) {
    mBufferPool->release(false);
}


CUDABackend::CUDABackend(std::shared_ptr<BufferAllocator> st,
                         std::shared_ptr<CUDARuntime> rt, int precision)
    : Backend(MNN_FORWARD_CUDA) {
#ifdef LOG_VERBOSE
        MNN_PRINT("cuda backend create\n");
#endif
    mBufferPool.reset(new BufferAllocator(BufferAllocator::Allocator::createRecurse(st.get())));
    mStaticBufferPool = st;
    mCUDARuntime      = rt;
    mUseFp16AsFp32 = (precision == 2);
    mPrecision = precision;
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
const Runtime* CUDABackend::getRuntime() {
    return (const Runtime*)mCUDARuntime.get();
}
bool CUDABackend::useFp16() const {
    return mUseFp16AsFp32;
}
int CUDABackend::getPrecision() const {
    return mPrecision;
}

class CUDAMemObj : public Backend::MemObj {
public:
    CUDAMemObj(BufferAllocator* allocator, std::pair<void*, int> points) {
        mPoint = std::move(points);
        mAllocator = allocator;
    }
    virtual ~ CUDAMemObj() {
        mAllocator->free(mPoint);
    }
private:
    BufferAllocator* mAllocator;
    std::pair<void*, int> mPoint;
};
int CUDABackend::getBytes(const Tensor* tensor) const {
    auto bytes = tensor->getType().bytes();
    if (mUseFp16AsFp32) {
        if (halide_type_float == tensor->getType().code) {
            bytes = 2;
        }
    }
    return bytes;
}
CPUResizeCache* CUDABackend::getCache() {
    return &mCache;
}

Backend::MemObj* CUDABackend::onAcquire(const Tensor* nativeTensor, StorageType storageType) {
    // MNN_PRINT("onAcquire CUDA memory for tensor:%p\n", nativeTensor);
#ifdef LOG_VERBOSE
    MNN_PRINT("Start CUDABackend::onAcquireBuffer !\n");
#endif
    BufferAllocator* allocator = nullptr;
    auto bytes = getBytes(nativeTensor);
    size_t mallocSize = realSize(nativeTensor) * bytes;

    std::pair<void*, int> buffer;
    if (storageType == DYNAMIC_SEPERATE) {
        buffer                              = mBufferPool->alloc(mallocSize, true);
        allocator = mBufferPool.get();
    } else if (storageType == DYNAMIC) {
        buffer                              = mBufferPool->alloc(mallocSize, false);
        allocator = mBufferPool.get();
    } else {
        MNN_ASSERT(storageType == STATIC);
        buffer                              = mStaticBufferPool->alloc(mallocSize, false);
        allocator = mStaticBufferPool.get();
    }
    if(nullptr == buffer.first) {
        return nullptr;
    };
    auto host = (uint8_t*)buffer.first + buffer.second;
    ((Tensor*)nativeTensor)->buffer().device = (uint64_t)host;
    auto des = TensorUtils::getDescribe(nativeTensor);
    des->extra.offset = buffer.second;
    return new CUDAMemObj(allocator, buffer);
}

bool CUDABackend::onClearBuffer() {
    mCache.reset();
    mBufferPool->release(true);
    return true;
}
size_t CUDABackend::realSize(const Tensor* tensor) {
    auto dim = TensorUtils::getDescribe(tensor)->dimensionFormat;
    int pack = 1;
    if (dim == MNN_DATA_FORMAT_NC4HW4) {
        pack = PACK_NUMBER;
    }
    size_t res = 1;
    for (int i = 0; i < tensor->dimensions(); ++i) {
        size_t l = tensor->length(i);
        if (1 == i ) {
            l = UP_DIV(l, pack) * pack;
        }
        res *= l;
    }
    return res;
}

Execution* CUDABackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const MNN::Op* op) {
// #ifdef LOG_VERBOSE
    // MNN_PRINT("Start CUDABackend::onCreate useFp16:%d\n", useFp16());
// #endif
    auto creators = gCreator();
    auto iter     = creators->find(op->type());
    if (iter == creators->end()) {
        if (nullptr != op->name()) {
            MNN_PRINT("CUDABackend Don't support type %s, %s\n", EnumNameOpType(op->type()), op->name()->c_str());
        } else {
            MNN_PRINT("CUDABackend Don't support type %s\n", EnumNameOpType(op->type()));
        }
        return NULL;
    }

    auto exe = iter->second->onCreate(inputs, outputs, op, this);
    if (NULL == exe) {
        if (nullptr != op->name()) {
            MNN_PRINT("CUDABackend The Creator Don't support type %s, %s\n", EnumNameOpType(op->type()), op->name()->c_str());
        } else {
            MNN_PRINT("CUDABackend The Creator Don't support type %s\n", EnumNameOpType(op->type()));
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

static void _computeStride(MNN_DATA_FORMAT srcDimensionFormat, int* srcStride, int batch, int plane, int channel, int srcPack) {
    if (srcDimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        srcStride[0] = plane * srcPack;
        srcStride[1] = plane * batch * PACK_NUMBER;
        srcStride[2] = srcPack;
    } else if (srcDimensionFormat == MNN_DATA_FORMAT_NCHW) {
        srcStride[0] = channel * plane;
        srcStride[1] = plane * PACK_NUMBER;
        srcStride[2] = 1;
    } else {
        srcStride[0] = channel * plane;
        srcStride[1] = PACK_NUMBER;
        srcStride[2] = channel;
    }
}

static void _computeBCA(int& batch, int& plane, int& channel, MNN_DATA_FORMAT srcDimensionFormat, const Tensor* srcTensor) {
    if(srcTensor->dimensions() == 0) {
        batch = 1;
        plane = 1;
        channel = 1;
        return;
    }

    if (srcDimensionFormat != MNN_DATA_FORMAT_NHWC) {
        batch = srcTensor->length(0);
        channel = srcTensor->length(1);
        plane = 1;
        for (int i=2; i<srcTensor->dimensions(); ++i) {
            plane *= srcTensor->length(i);
        }
    } else {
        batch = srcTensor->length(0);
        channel = 1;
        if(srcTensor->dimensions() > 1) {
            channel = srcTensor->length(srcTensor->dimensions()-1);
        }
        plane = 1;
        for (int i=1; i<srcTensor->dimensions()-1; ++i) {
            plane *= srcTensor->length(i);
        }
    }
}

static PackInfo _computePackInfo(MNN_DATA_FORMAT srcDimensionFormat, int batch, int plane, int channel) {
    PackInfo pack;
    pack.inside = plane;
    pack.axis = channel;
    pack.unit = PACK_NUMBER;
    pack.outside = batch;
    if (srcDimensionFormat == MNN_DATA_FORMAT_NHWC) {
        pack.axisStride = 1;
        pack.insideStride = channel;
    } else {
        pack.axisStride = plane;
        pack.insideStride = 1;
    }
    return pack;
}

void CUDABackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {

    auto srcDimensionFormat = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
    auto dstDimensionFormat = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
    auto srcIndex = TensorUtils::getDescribe(srcTensor)->index;
    auto dstIndex = TensorUtils::getDescribe(dstTensor)->index;
    auto srcDevice          = srcTensor->deviceId() != 0;
    auto dstDevice          = dstTensor->deviceId() != 0;
    MNN_ASSERT(srcDevice || dstDevice);
    uint8_t* srcPtr = nullptr;
    std::pair<void*, int> tempSrcStorage;
    auto bytes = getBytes(srcTensor);
    auto type = srcTensor->getType();

    //printf("%d-%d\n", srcTensor->dimensions(), dstTensor->dimensions());
    bool directCopy = (srcDimensionFormat == dstDimensionFormat && dstDimensionFormat != MNN_DATA_FORMAT_NC4HW4) || srcTensor->dimensions() <= 1;
    if (mUseFp16AsFp32) {
        if (((!srcDevice) || (!dstDevice))){
            if (type.code == halide_type_float) {
                directCopy = false;
            }
        }
    }

#ifdef MNN_CUDA_COPY_DEBUG
    checkKernelErrors;
    MNN_PRINT("CUDA Bn copy tensor ptr:%p -> ptr:%p deviceId:%d -> %d, hostPtr:%p -> %p, graphIndex: %d -> %d, format %d -> %d, directCopy: %d, dims: [",
        srcTensor, dstTensor, srcTensor->deviceId(), dstTensor->deviceId(), srcTensor->host<void>(), dstTensor->host<void>(), srcIndex, dstIndex, srcDimensionFormat, dstDimensionFormat, directCopy);

    for (int i=0; i<srcTensor->dimensions(); ++i) {
        MNN_PRINT("%d ", srcTensor->length(i));
        if(srcDevice && !dstDevice) {
            printf("\n");
        }
    }
    MNN_PRINT("], ");
    MNN_PRINT("addr:%p %p\n", srcTensor->deviceId(), dstTensor->deviceId());
#endif


    if (directCopy) {
        auto gpuSize = realSize(srcTensor) * getBytes(srcTensor);
        if (srcDevice && dstDevice) {
            NVTX_PUSH("DtoD");
            mCUDARuntime->memcpy((void*)(dstTensor->deviceId()), (void*)(srcTensor->deviceId()), gpuSize,
                                MNNMemcpyDeviceToDevice, true);
            NVTX_POP();
        } else if (srcDevice && (!dstDevice)) {
            NVTX_PUSH("DtoH");
            mCUDARuntime->memcpy((void*)(dstTensor->host<void>()), (void*)(srcTensor->deviceId()), gpuSize,
                                MNNMemcpyDeviceToHost, true);
            NVTX_POP();
        } else if ((!srcDevice) && (dstDevice)) {
            NVTX_PUSH("HtoD");
            mCUDARuntime->memcpy((void*)(dstTensor->deviceId()), (void*)(srcTensor->host<void>()), gpuSize,
                                MNNMemcpyHostToDevice, true);
            NVTX_POP();
        }
        return;
    }
    if (!srcDevice) {
        auto cpuSize = srcTensor->size();
        tempSrcStorage = mStaticBufferPool->alloc(cpuSize);
        srcPtr = (uint8_t*)tempSrcStorage.first + tempSrcStorage.second;
        mCUDARuntime->memcpy(srcPtr, srcTensor->host<void>(), cpuSize, MNNMemcpyHostToDevice,
                             true);
    } else {
        srcPtr = (uint8_t*)srcTensor->deviceId();
    }
    uint8_t* dstPtr = nullptr;
    std::pair<void*, int> tempDstStorage;
    if (!dstDevice) {
        auto cpuSize = dstTensor->size();
        tempDstStorage = mStaticBufferPool->alloc(cpuSize);
        dstPtr = (uint8_t*)tempDstStorage.first + tempDstStorage.second;
    } else {
        dstPtr = (uint8_t*)dstTensor->deviceId();
    }

    NVTX_PUSH("copy convert");
    // Format convert
    int batch, plane, channel;
    _computeBCA(batch, plane, channel, srcDimensionFormat, srcTensor);

    // for (int i=0; i<srcTensor->dimensions(); ++i) {
    //     MNN_PRINT("%d ", srcTensor->length(i));
    // }
    // MNN_PRINT("\n, batch:%d, plane:%d, channel:%d, dims:%d\n", batch, plane, channel, srcTensor->dimensions());

    FormatConvert((float *)dstPtr, (float *)srcPtr, srcDimensionFormat, dstDimensionFormat, mCUDARuntime.get(), \
            plane, batch, channel, srcTensor, \
            mUseFp16AsFp32, srcDevice, dstDevice);

    if (!srcDevice) {
        mStaticBufferPool->free(tempSrcStorage);
    }
    if (!dstDevice) {
        auto cpuSize = dstTensor->size();
        mCUDARuntime->memcpy(dstTensor->host<void>(), dstPtr, cpuSize, MNNMemcpyDeviceToHost,
                             true);
        mStaticBufferPool->free(tempDstStorage);
    }
    NVTX_POP();
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
