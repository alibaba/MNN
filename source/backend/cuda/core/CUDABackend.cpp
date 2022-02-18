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
    auto mode = mDefaultPrecision;
    if (nullptr != config) {
        mode = config->precision;
    }
    bool useFp16 = mode == BackendConfig::Precision_Low;
    return new CUDABackend(mBufferPool, mCUDARuntime, useFp16);
}

void CUDARuntimeWrapper::onGabageCollect(int level) {
    mBufferPool->release(false);
}

CUDABackend::CUDABackend(std::shared_ptr<BufferAllocator> st,
                         std::shared_ptr<CUDARuntime> rt, bool useFp16AsFp32)
    : Backend(MNN_FORWARD_CUDA) {
    mBufferPool.reset(new BufferAllocator(BufferAllocator::Allocator::createRecurse(st.get())));
    mStaticBufferPool = st;
    mCUDARuntime      = rt;
    mUseFp16AsFp32 = useFp16AsFp32;
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
bool CUDABackend::useFp16() const {
    return mUseFp16AsFp32;
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
    if (srcDimensionFormat != MNN_DATA_FORMAT_NHWC) {
        batch = srcTensor->length(0);
        channel = srcTensor->length(1);
        plane = 1;
        for (int i=2; i<srcTensor->dimensions(); ++i) {
            plane *= srcTensor->length(i);
        }
    } else {
        batch = srcTensor->length(0);
        channel = srcTensor->length(srcTensor->dimensions()-1);
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
    auto srcDevice          = srcTensor->deviceId() != 0;
    auto dstDevice          = dstTensor->deviceId() != 0;
    MNN_ASSERT(srcDevice || dstDevice);
    uint8_t* srcPtr = nullptr;
    std::pair<void*, int> tempSrcStorage;
    auto bytes = getBytes(srcTensor);
    auto type = srcTensor->getType();
#ifdef MNN_CUDA_COPY_DEBUG
    MNN_PRINT("CUDA Bn copy: %d -> %d, format %d -> %d, dims: [", srcDevice, dstDevice, srcDimensionFormat, dstDimensionFormat);
    for (int i=0; i<srcTensor->dimensions(); ++i) {
        MNN_PRINT("%d ", srcTensor->length(i));
    }
    MNN_PRINT("]\n");
#endif
    bool directCopy = (srcDimensionFormat == dstDimensionFormat && dstDimensionFormat != MNN_DATA_FORMAT_NC4HW4) || srcTensor->dimensions() <= 1;
    if (mUseFp16AsFp32) {
        if ((!srcDevice) || (!dstDevice)) {
            if (type.code == halide_type_float) {
                directCopy = false;
            }
        }
    }
    if (directCopy) {
        auto gpuSize = realSize(srcTensor) * getBytes(srcTensor);
        if (srcDevice && dstDevice) {
            mCUDARuntime->memcpy((void*)(dstTensor->deviceId()), (void*)(srcTensor->deviceId()), gpuSize,
                                MNNMemcpyDeviceToDevice, true);
        } else if (srcDevice && (!dstDevice)) {
            mCUDARuntime->memcpy((void*)(dstTensor->host<void>()), (void*)(srcTensor->deviceId()), gpuSize,
                                MNNMemcpyDeviceToHost, true);
        } else if ((!srcDevice) && (dstDevice)) {
            mCUDARuntime->memcpy((void*)(dstTensor->deviceId()), (void*)(srcTensor->host<void>()), gpuSize,
                                MNNMemcpyHostToDevice, true);
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

    // Format convert
    FuseRegion reg;
    int* size = reg.size;
    int* srcStride = reg.srcStride;
    int* dstStride = reg.dstStride;
    int offset[PACK_NUMBER * 8];
    int offsetNumber = 0;
    auto offsetGpuStorage = mStaticBufferPool->alloc(PACK_NUMBER * 8 * sizeof(int));
    auto offsetGpu = (uint8_t*)offsetGpuStorage.first + offsetGpuStorage.second;
    auto regionStorage = mStaticBufferPool->alloc(sizeof(FuseRegion));
    auto regionGpu = (FuseRegion*)((uint8_t*)regionStorage.first + regionStorage.second);

    do {
        if (srcTensor->deviceId() != 0 && dstTensor->deviceId() != 0) {
            if (srcTensor->dimensions() <= 1 || srcDimensionFormat == dstDimensionFormat) {
                auto gpuSize = realSize(srcTensor) * getBytes(srcTensor);
                mCUDARuntime->memcpy((void*)(dstTensor->deviceId()), (void*)(srcTensor->deviceId()), gpuSize,
                                    MNNMemcpyDeviceToDevice, true);
            } else {
                int batch, plane, channel;
                _computeBCA(batch, plane, channel, srcDimensionFormat, srcTensor);
                PackInfo pack;
                auto func = PackBuffer;
                if (dstDimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
                    pack = _computePackInfo(srcDimensionFormat, batch, plane, channel);
                    func = PackBuffer;
                } else if (srcDimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
                    pack = _computePackInfo(dstDimensionFormat, batch, plane, channel);
                    func = UnpackBuffer;
                } else {
                    FUNC_PRINT(1);
                }
                func((void*)(dstTensor->deviceId()), (void*)(srcTensor->deviceId()), &pack, getBytes(srcTensor), mCUDARuntime.get());
            }
            break;
        }
        auto convertFunction = FuseRasterBlitFloatToFloat;
        if (mUseFp16AsFp32) {
            if (!srcDevice) {
                convertFunction = FuseRasterBlitFloatToHalf;
            } else {
                convertFunction = FuseRasterBlitHalfToFloat;
            }
        }
        if (srcTensor->dimensions() <= 1) {
            size[2] = srcTensor->elementSize();
            srcStride[2] = 1;
            dstStride[2] = 1;
            offset[0] = 1;
            offset[1] = 1;
            offset[2] = size[2];
            offset[3] = 0;
            offset[4] = 1;
            offset[5] = 1;
            offset[6] = size[2];
            offset[7] = 0;
            offsetNumber = 1;
        } else {
            // Compute batch, plane, channel
            int batch, plane, channel;
            _computeBCA(batch, plane, channel, srcDimensionFormat, srcTensor);
            if (dstDimensionFormat == MNN_DATA_FORMAT_NC4HW4 && srcDimensionFormat != MNN_DATA_FORMAT_NC4HW4 && dstDevice) {
                PackInfo pack = _computePackInfo(srcDimensionFormat, batch, plane, channel);
                if (mUseFp16AsFp32) {
                    if (type.code == halide_type_float) {
                        if (dstDevice) {
                            PackFP32ToFP16(dstPtr, srcPtr, &pack, mCUDARuntime.get());
                            break;
                        } else {
                            PackFP16ToFP32(dstPtr, srcPtr, &pack, mCUDARuntime.get());
                            break;
                        }
                    }
                } else {
                    PackBuffer(dstPtr, srcPtr, &pack, bytes, mCUDARuntime.get());
                }
                break;
            }
            if (srcDimensionFormat == MNN_DATA_FORMAT_NC4HW4 && dstDimensionFormat != MNN_DATA_FORMAT_NC4HW4 && srcDevice) {
                PackInfo pack = _computePackInfo(dstDimensionFormat, batch, plane, channel);
                if (mUseFp16AsFp32) {
                    if (type.code == halide_type_float) {
                        if (dstDevice) {
                            UnpackFP32ToFP16(dstPtr, srcPtr, &pack, mCUDARuntime.get());
                            break;
                        } else {
                            UnpackFP16ToFP32(dstPtr, srcPtr, &pack, mCUDARuntime.get());
                            break;
                        }
                    }
                } else {
                    UnpackBuffer(dstPtr, srcPtr, &pack, bytes, mCUDARuntime.get());
                }
                break;
            }
            //MNN_PRINT("host/device: %d -> %d, format %d -> %d, b, p, c: %d - %d - %d\n", srcDevice, dstDevice, srcDimensionFormat, dstDimensionFormat, batch, plane, channel);
            // Set region
            if (srcDimensionFormat != MNN_DATA_FORMAT_NC4HW4 && dstDimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                size[0] = batch;
                size[1] = channel;
                size[2] = plane;
                offsetNumber = 1;
                offset[0] = batch;
                offset[1] = channel;
                offset[2] = plane;
                offset[3] = 0;
                offset[4] = batch;
                offset[5] = channel;
                offset[6] = plane;
                offset[7] = 0;
                if (srcDimensionFormat == MNN_DATA_FORMAT_NHWC) {
                    srcStride[0] = channel * plane;
                    srcStride[1] = 1;
                    srcStride[2] = channel;
                } else {
                    srcStride[0] = channel * plane;
                    srcStride[1] = plane;
                    srcStride[2] = 1;
                }
                if (dstDimensionFormat == MNN_DATA_FORMAT_NHWC) {
                    dstStride[0] = channel * plane;
                    dstStride[1] = 1;
                    dstStride[2] = channel;
                } else {
                    dstStride[0] = channel * plane;
                    dstStride[1] = plane;
                    dstStride[2] = 1;
                }
            } else {
                offsetNumber = PACK_NUMBER;
                size[0] = batch;
                size[1] = UP_DIV(channel, PACK_NUMBER);
                size[2] = plane;
                int srcPack = 1;
                int dstPack = 1;
                int srcChannelLimit = channel;
                if (srcDimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
                    if (srcDevice) {
                        srcPack = PACK_NUMBER;
                        srcChannelLimit = UP_DIV(channel, PACK_NUMBER) * PACK_NUMBER;
                    } else {
                        srcPack = 4;
                        srcChannelLimit = UP_DIV(channel, 4) * 4;
                    }
                }
                int dstChannelLimit = channel;
                if (dstDimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
                    if (dstDevice) {
                        dstPack = PACK_NUMBER;
                        dstChannelLimit = UP_DIV(channel, PACK_NUMBER) * PACK_NUMBER;
                    } else {
                        dstPack = 4;
                        dstChannelLimit = UP_DIV(channel, 4) * 4;
                    }
                }
                // Compute Stride
                _computeStride(srcDimensionFormat, srcStride, batch, plane, channel, srcPack);
                _computeStride(dstDimensionFormat, dstStride, batch, plane, channel, dstPack);

                // Compute Offset
                for (int i=0; i<offsetNumber; ++i) {
                    auto offsetPtr = offset + i * 8;
                    int channelStart = i;
                    offsetPtr[0] = batch;
                    offsetPtr[1] = (srcChannelLimit + PACK_NUMBER - channelStart - 1) / PACK_NUMBER;
                    offsetPtr[2] = plane;
                    if (srcDimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
                        int sp = i / srcPack;
                        int sm = i % srcPack;
                        offsetPtr[3] = sm + sp * srcPack * plane * batch;
                    } else {
                        offsetPtr[3] = channelStart * srcStride[1] / PACK_NUMBER;
                    }

                    offsetPtr[4] = batch;
                    offsetPtr[5] = (dstChannelLimit + PACK_NUMBER - channelStart - 1) / PACK_NUMBER;
                    offsetPtr[6] = plane;
                    if (dstDimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
                        int sp = i / dstPack;
                        int sm = i % dstPack;
                        offsetPtr[7] = sm + sp * dstPack * plane * batch;
                    } else {
                        offsetPtr[7] = channelStart * dstStride[1] / PACK_NUMBER;
                    }
                }
            }
        }
        reg.fuseNumber = offsetNumber;
        mCUDARuntime->memcpy(regionGpu, &reg, sizeof(FuseRegion), MNNMemcpyHostToDevice, true);
        mCUDARuntime->memcpy(offsetGpu, offset, offsetNumber * 8 * sizeof(int), MNNMemcpyHostToDevice, true);
#ifdef MNN_CUDA_COPY_DEBUG
        MNN_PRINT("Reg.size: %d - %d - %d\n", reg.size[0], reg.size[1], reg.size[2]);
        MNN_PRINT("Reg.srcStride: %d - %d - %d\n", reg.srcStride[0], reg.srcStride[1], reg.srcStride[2]);
        MNN_PRINT("Reg.dstStride: %d - %d - %d\n", reg.dstStride[0], reg.dstStride[1], reg.dstStride[2]);
        MNN_PRINT("FuseNum: %d\n", reg.fuseNumber);
        for (int i=0; i<reg.fuseNumber; ++i) {
            auto off = offset + 8 * i;
            MNN_PRINT("Src: %d, %d, %d, %d; dst:%d, %d, %d, %d\n", off[0], off[1], off[2], off[3], off[4], off[5], off[6], off[7]);
        }
#endif
        if (mUseFp16AsFp32) {
            if (type.code == halide_type_float) {
                convertFunction(dstPtr, srcPtr, regionGpu, offsetGpu, mCUDARuntime.get());
                break;
            }
        }
        FuseRasterBlitCommon(dstPtr, srcPtr, regionGpu, offsetGpu, mCUDARuntime.get(), type.bytes());
    } while(false);
    mStaticBufferPool->free(offsetGpuStorage);
    mStaticBufferPool->free(regionStorage);
    if (!srcDevice) {
        mStaticBufferPool->free(tempSrcStorage);
    }
    if (!dstDevice) {
        auto cpuSize = dstTensor->size();
        mCUDARuntime->memcpy(dstTensor->host<void>(), dstPtr, cpuSize, MNNMemcpyDeviceToHost,
                             true);
        mStaticBufferPool->free(tempDstStorage);        
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
