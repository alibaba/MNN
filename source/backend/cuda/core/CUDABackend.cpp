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
#include "execution/CastExecution.hpp"
#include "CUDATools.hpp"
#include "execution/FuseExecutionV2.hpp"
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
    virtual MemChunk onAlloc(size_t size, size_t align) override {
        return MemChunk(mRuntime->alloc(size), 0);
    }
    virtual void onRelease(MemChunk ptr) override {
        mRuntime->free(ptr.first);
    }
private:
    CUDARuntime* mRuntime;
};
CUDARuntimeWrapper::CUDARuntimeWrapper(BackendConfig::PrecisionMode precision, BackendConfig::PowerMode power, BackendConfig::MemoryMode memory, int deviceId) {
    // TODO: Search CUDA Device info and use best one
    mCUDARuntime.reset(new CUDARuntime(deviceId));
#ifdef LOG_VERBOSE
    MNN_PRINT("create cuda runtime:%p\n", mCUDARuntime.get());
#endif
    if (mCUDARuntime.get()) {
        if (mCUDARuntime->isCreateError() == true) {
            mIsCreateError = true;
            return;
        }
        std::shared_ptr<BufferAllocator::Allocator> allocator(new CUDARuntimeAllocator(mCUDARuntime.get()));
        mBufferPool.reset(new EagerBufferAllocator(allocator));
    }
    mDefaultPrecision = precision;
    mDefaultMemory = memory;
}
CUDARuntimeWrapper::~CUDARuntimeWrapper() {
    // Do nothing
}
float CUDARuntimeWrapper::onGetMemoryInMB() {
    auto staticMemoryInMB = mBufferPool->totalSize() / 1024.0f / 1024.0f;
    return staticMemoryInMB;
}

std::pair<const void*, size_t> CUDARuntimeWrapper::onGetCache() {//make Cache
    return mCUDARuntime->makeCache();
}

bool CUDARuntimeWrapper::onSetCache(const void* buffer, size_t size) {//set Cache
    return mCUDARuntime->setCache(std::make_pair(buffer, size));
}

Backend* CUDARuntimeWrapper::onCreate(const BackendConfig* config) const {
#ifdef LOG_VERBOSE
    MNN_PRINT("cudaruntime:%p, create CUDABackend\n", this);
#endif
    auto precision_mode = mDefaultPrecision;
    auto memory_mode = mDefaultMemory;
    if (nullptr != config) {
        precision_mode = config->precision;
        memory_mode = config->memory;
    }
    int precision = 0; 
    if(precision_mode == BackendConfig::Precision_Low) {
        precision = 2;
    } else if(precision_mode == BackendConfig::Precision_Normal) {
        precision = 0;
    } else if(precision_mode == BackendConfig::Precision_Low_BF16) {
        precision = 3;
    } else {
        precision = 1;
    }

    return new CUDABackend(mBufferPool, mCUDARuntime, precision, memory_mode);
}

void CUDARuntimeWrapper::onGabageCollect(int level) {
    mBufferPool->release(false);
}


CUDABackend::CUDABackend(std::shared_ptr<BufferAllocator> st,
                         std::shared_ptr<CUDARuntime> rt,
                        int precision, BackendConfig::MemoryMode memory)
    : Backend(MNN_FORWARD_CUDA) {
#ifdef LOG_VERBOSE
        MNN_PRINT("cuda backend create\n");
#endif
    mBufferPool.reset(new EagerBufferAllocator(BufferAllocator::Allocator::createRecurse(st.get())));
    mStaticBufferPool = st;
    mCUDARuntime      = rt;
    mUseFp16AsFp32 = (precision == 2);
    mPrecision = precision;
    mMemory = memory;
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

#ifdef MNN_CODEGEN_CUDA
std::map<std::pair<std::string, std:: string>, CUmodule> CUDABackend::kernelCuModuleMap() {
    return mKernelCuModuleMap;
}
#endif

int CUDABackend::getPrecision() const {
    return mPrecision;
}

BackendConfig::MemoryMode CUDABackend::getMemoryMode() const {
    return mMemory;
}
class CUDAMemObj : public Backend::MemObj {
public:
    CUDAMemObj(BufferAllocator* allocator, MemChunk points) {
        mPoint = std::move(points);
        mAllocator = allocator;
    }
    virtual ~ CUDAMemObj() {
        mAllocator->free(mPoint);
    }
    MemChunk chunk() override {
        return mPoint;
    }
private:
    BufferAllocator* mAllocator;
    MemChunk mPoint;
};
int CUDABackend::getBytes(const Tensor* tensor) const {
    auto bytes = tensor->getType().bytes();
    if (mPrecision == 2 || mPrecision == 3) {// Fp16 or Bf16
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

    MemChunk buffer;
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
    auto host = buffer.ptr();
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
        if (getDataType(tensor) == DataType_DT_INT8 || tensor->getType().bytes() == 1) {
            pack = INT8_PACK_NUMBER;
        }
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

static OpType _getRealOpType(OpType opType) {
    switch (opType) {
        case OpType_Convolution:
            return OpType_ConvInt8;
        case OpType_ConvolutionDepthwise:
            return OpType_DepthwiseConvInt8;
        case OpType_BinaryOp:
        default:
            return opType;
    }
}

#ifdef MNN_CODEGEN_CUDA
void CUDABackend::compile(CUmodule* dst, std::pair<string, string> code, std::vector<const char*> compile_params) {
    std::vector<const char *> param;
    auto ptx_code =
        CUDANVRTCCompile(code, param, mCUDARuntime->compute_capability(), false);

    MNN_CUDA_SAFE_CALL(cuModuleLoadDataEx(dst, ptx_code.c_str(), 0, 0, 0));
}
#endif

Execution* CUDABackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const MNN::Op* op) {
// #ifdef LOG_VERBOSE
    // MNN_PRINT("Start CUDABackend::onCreate useFp16:%d\n", useFp16());
// #endif
    auto opType = op->type();
    if (outputs.size() > 0) {
        if (TensorUtils::getDescribe(outputs[0])->quantAttr != nullptr && TensorUtils::getDescribe(outputs[0])->type == DataType_DT_INT8) {
            opType = _getRealOpType(opType);
        }
    }
    // MNN_PRINT("CUDABackend support type %s\n", EnumNameOpType(opType));
    auto creators = gCreator();
    auto iter     = creators->find(opType);
    if (iter == creators->end()) {
        if (nullptr != op->name()) {
            MNN_PRINT("CUDABackend Don't support type %s, %s\n", EnumNameOpType(opType), op->name()->c_str());
        } else {
            MNN_PRINT("CUDABackend Don't support type %s\n", EnumNameOpType(opType));
        }
        return NULL;
    }

    #ifdef MNN_CODEGEN_CUDA
    if(op->type() == OpType_Extra) {
        if (!FuseExecutionV2::check(op)) {
            auto extra = op->main_as_Extra();
            std::string source(reinterpret_cast<const char*>(extra->info()->data()));
            auto kernel_name = extra->type()->c_str();
            std::string kernel_source = source;

            std::pair<std::string, std::string> kernelInfo = std::make_pair<std::string, std::string>(kernel_name, kernel_source.c_str());
            if(mKernelCuModuleMap.find(kernelInfo) == mKernelCuModuleMap.end()) {
                // printf("\n%s\n\n%s !!!!\n", kernel_source.c_str(), kernel_name);
                std::vector<const char *> param;
                bool includeHeadFile = mUseFp16AsFp32;
                auto ptx_code =
                    CUDANVRTCCompile(kernelInfo, param, mCUDARuntime->compute_capability(), includeHeadFile);
                
                MNN_CUDA_SAFE_CALL(cuModuleLoadDataEx(&mCuModule, ptx_code.c_str(), 0, 0, 0));
                mKernelCuModuleMap.insert(std::pair<std::pair<std::string, std:: string>, CUmodule>(kernelInfo, mCuModule));
            }
        }
    }
    #endif

    auto exe = iter->second->onCreate(inputs, outputs, op, this);
    if (NULL == exe) {
        if (nullptr != op->name()) {
            MNN_PRINT("CUDABackend The Creator Don't support type %s, %s\n", EnumNameOpType(opType), op->name()->c_str());
        } else {
            MNN_PRINT("CUDABackend The Creator Don't support type %s\n", EnumNameOpType(opType));
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

ErrorCode CUDABackend::onResizeEnd() {
    return NO_ERROR;
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
        channel = 1;
        if(srcTensor->dimensions() > 1) {
            channel = srcTensor->length(1);
        }
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
    auto srcDevice = (srcTensor->deviceId() != 0 && srcTensor->deviceId() != 1);
    auto dstDevice = (dstTensor->deviceId() != 0 && dstTensor->deviceId() != 1);    
    MNN_ASSERT(srcDevice || dstDevice);
    uint8_t* srcPtr = nullptr;
    MemChunk tempSrcStorage;
    auto bytes = getBytes(srcTensor);
    auto type = srcTensor->getType();

    //MNN_PRINT("%d-%d\n", srcTensor->dimensions(), dstTensor->dimensions());
    bool directCopy = ((srcDimensionFormat == dstDimensionFormat && dstDimensionFormat != MNN_DATA_FORMAT_NC4HW4) || srcTensor->dimensions() <= 1) && \
        (getDataType(srcTensor) == getDataType(dstTensor));
    if (mPrecision == 2 || mPrecision == 3) { // Fp16 or Bf16
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
            MNN_PRINT("\n");
        }
    }
    MNN_PRINT("], ");
    MNN_PRINT("addr:%p %p\n", srcTensor->deviceId(), dstTensor->deviceId());
#endif

    // printf("MNN srcDevice:%d %llu, dstDevice:%d %llu, directCopy:%d\n", srcDevice, srcTensor->deviceId(), dstDevice, dstTensor->deviceId(), directCopy);
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
        srcPtr = tempSrcStorage.ptr();
        mCUDARuntime->memcpy(srcPtr, srcTensor->host<void>(), cpuSize, MNNMemcpyHostToDevice,
                             true);
    } else {
        srcPtr = (uint8_t*)srcTensor->deviceId();
    }
    uint8_t* dstPtr = nullptr;
    MemChunk tempDstStorage;
    if (!dstDevice) {
        auto cpuSize = dstTensor->size();
        tempDstStorage = mStaticBufferPool->alloc(cpuSize);
        dstPtr = tempDstStorage.ptr();
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
    // MNN_PRINT("oncopybuffer dateType:%d->%d  format:%d->%d\n", getDataType(srcTensor), getDataType(dstTensor), srcDimensionFormat, dstDimensionFormat);

    std::unique_ptr<Tensor> wrapTensor;
    MemChunk wrapSrcStorage;
    if (getDataType(srcTensor) != getDataType(dstTensor)) {
        auto dimType = Tensor::CAFFE;
        switch (TensorUtils::getDescribe(srcTensor)->dimensionFormat) {
            case MNN_DATA_FORMAT_NCHW:
                break;
            case MNN_DATA_FORMAT_NC4HW4:
                dimType = Tensor::CAFFE_C4;
                break;
            case MNN_DATA_FORMAT_NHWC:
                dimType = Tensor::TENSORFLOW;
                break;
            default:
                break;
        }

        auto convertType = CastCreator::FlOAT_TO_INT8;
        if (getDataType(srcTensor) == DataType_DT_INT8) {
            convertType = CastCreator::INT8_TO_FlOAT;
        }

        wrapTensor.reset(Tensor::createDevice(srcTensor->shape(), dstTensor->getType(), dimType));
        wrapSrcStorage = mStaticBufferPool->alloc(realSize(wrapTensor.get()) * getBytes(dstTensor));
        // MNN_PRINT("warp:%d %d %d %d\n", realSize(wrapTensor.get()), getBytes(dstTensor), dstTensor->getType(), srcTensor->getDimensionType());
        wrapTensor.get()->buffer().device = (uint64_t)(wrapSrcStorage.ptr());

        auto dstType = getDataType(dstTensor);
        if (dstType != DataType_DT_FLOAT) {
            wrapTensor->setType(dstType);
        }

#ifdef LOG_VERBOSE
        MNN_PRINT("CPU backend copy tensor ptr:%p -> ptr:%p hostPtr:%p -> %p, format %d -> %d, dims: [",
        srcTensor, dstTensor, srcTensor->host<void>(), dstTensor->host<void>(), TensorUtils::getDescribe(srcTensor)->dimensionFormat, TensorUtils::getDescribe(dstTensor)->dimensionFormat);
        for (int i=0; i<srcTensor->dimensions(); ++i) {
            MNN_PRINT("%d ", srcTensor->length(i));
        }
        MNN_PRINT("]\n");
#endif

        auto code = CastCreator::cast(srcTensor, wrapTensor.get(), (Backend*)this, convertType);
        if (NO_ERROR != code) {
            MNN_ERROR("Error in CudaBackend::onCopyBuffer:cast\n");
        }
        srcTensor = wrapTensor.get();
        srcPtr = (uint8_t*)srcTensor->deviceId();
    }

    FormatConvert((float *)dstPtr, (float *)srcPtr, srcDimensionFormat, dstDimensionFormat, mCUDARuntime.get(), \
            plane, batch, channel, srcTensor, \
            mPrecision, srcDevice, dstDevice);

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

int CUDABackend::onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) {
    if (toCpu) {
        mCUDARuntime->device_sync();
    }
    return 0;
}

DataType CUDABackend::getDataType(const Tensor* tensor) {
    auto des = TensorUtils::getDescribe(tensor);
    if (nullptr == des->quantAttr.get()) {
        return DataType_DT_FLOAT;
    }
    return des->type;
}

ErrorCode CastWrapExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto convertType = mRunType == DataType_DT_INT8 ? CastCreator::FlOAT_TO_INT8 : CastCreator::INT8_TO_FlOAT;
    auto cudaBackend = ((CUDABackend*)backend());
    CastCreator::cast(inputs[0], outputs[0], cudaBackend, convertType);
    return NO_ERROR;
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
