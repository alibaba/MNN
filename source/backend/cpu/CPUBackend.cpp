//
//  CPUBackend.cpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUBackend.hpp"
#include <cmath>
#include <mutex>
#include "CPUResizeCache.hpp"
#include "core/BufferAllocator.hpp"
#include "CPUTensorConvert.hpp"
#include "compute/CommonOptFunction.h"
#include "core/TensorUtils.hpp"
#include "ThreadPool.hpp"
#include "core/Concurrency.h"
#include "CPUCast.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/WrapExecution.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP
#include "backend/cpu/CPURuntime.hpp"
#include "core/Macro.h"
#ifdef MNN_USE_ARMV82
#include "backend/arm82/Arm82Backend.hpp"
#endif
#define MAX_THREAD_NUMBER 32
#define LARGE_MEMORY 1024 * 1024 * 500
#ifdef MNN_SUPPORT_BF16
#include "bf16/BF16Backend.hpp"
#include "bf16/BF16Functions.hpp"
#endif

#ifdef MNN_USE_SSE
#include "x86_x64/AVX2Backend.hpp"
#endif

#define MNN_CPU_CHECK_NAN 1
#define MNN_CPU_USE_DEFAULT_BACKEND 4
namespace MNN {
void registerCPUOps();
ErrorCode CastWrapExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto convertType = mRunType == DataType_DT_INT8 ? CPUCastCreator::FlOAT_TO_INT8 : CPUCastCreator::INT8_TO_FlOAT;
    auto cpuBackend = ((CPUBackend*)backend());
    CPUCastCreator::cast(inputs[0], outputs[0], cpuBackend, convertType);
    return NO_ERROR;
}

CPURuntime::CPURuntime(const Backend::Info& info) {
    mStaticAllocator.reset(new EagerBufferAllocator(BufferAllocator::Allocator::createDefault()));
    mThreadNumber = info.numThread;
    mThreadNumber = std::max(1, mThreadNumber);
    mThreadNumber = std::min(mThreadNumber, MAX_THREAD_NUMBER);
    mPower   = BackendConfig::Power_Normal;
    mMemory  = BackendConfig::Memory_Normal;
    mPrecision = BackendConfig::Precision_Normal;
    mFlops = MNNGetCPUFlops(mThreadNumber);
    if (info.user != nullptr) {
        mPrecision = info.user->precision;
        mPower = info.user->power;
        mMemory = info.user->memory;
        mFlags = info.user->flags;
    }
    mAllocator = info.allocator;

#ifdef _OPENMP
    switch (mPower) {
        case BackendConfig::Power_Low:
            MNNSetCPUThreadsMode(MNN_CPU_MODE_LITTLE);
            break;
        case BackendConfig::Power_High:
            MNNSetCPUThreadsMode(MNN_CPU_MODE_POWER_FRI);
            break;
        default:
            break;
    }
#endif
#ifdef MNN_USE_THREAD_POOL
    mThreadNumber = ThreadPool::init(mThreadNumber);
    if (mThreadNumber > 1) {
        mTaskIndex = ThreadPool::acquireWorkIndex();
    } else {
        mTaskIndex = -1;
    }
    if (mTaskIndex >= 0 && mPower == BackendConfig::Power_High) {
        ThreadPool::active();
    }
#endif
#ifdef LOG_VERBOSE
    MNN_PRINT("create CPURuntime:%p\n", this);
#endif
}
CPURuntime:: ~ CPURuntime() {
#ifdef MNN_USE_THREAD_POOL
    if (mTaskIndex >= 0 && mPower == BackendConfig::Power_High) {
        ThreadPool::deactive();
    }
    ThreadPool::releaseWorkIndex(mTaskIndex);
#endif
}
float CPURuntime::onGetMemoryInMB() {
    auto staticMemoryInMB = mStaticAllocator->totalSize() / 1024.0f / 1024.0f;
    return staticMemoryInMB;
}
bool CPURuntime::onCheckInfo(Backend::Info& info) const {
#ifdef MNN_USE_THREAD_POOL
    int threadNumber = mThreadNumber;
    if (mTaskIndex < 0) {
        threadNumber = 1;
    }
    info.numThread = threadNumber;
#endif
    return true;
}

Backend* CPURuntime::onCreate(const BackendConfig* config) const {
    auto precision = mPrecision;
    auto memory = mMemory;
    size_t flags = mFlags;
    if (nullptr != config) {
        precision = config->precision;
        flags = config->flags;
        memory = config->memory;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("cpu backend was created by runtime:%p\n", this);
#endif

#ifdef MNN_USE_ARMV82
    auto core = MNNGetCoreFunctions();
    if (core->supportFp16arith && precision == BackendConfig::Precision_Low) {
        return new Arm82Backend(this, memory);
    }
#endif
#ifdef MNN_SUPPORT_BF16
    if (precision == BackendConfig::Precision_Low_BF16 && BF16Functions::get()) {
        return new BF16Backend(this);
    }
#endif
    if (flags == MNN_CPU_USE_DEFAULT_BACKEND) {
        return new CPUBackend(this, precision, memory, MNN_FORWARD_CPU, 0);
    }
#ifdef MNN_USE_SSE
    if (AVX2Backend::isValid()) {
        return new AVX2Backend(this, memory, flags);
    }
#endif

    return new CPUBackend(this, precision, memory, MNN_FORWARD_CPU, flags);
}

int CPURuntime::onGetRuntimeStatus(RuntimeStatus statusEnum) const {
    switch (statusEnum) {
        case STATUS_SUPPORT_FP16: {
            return MNNGetCoreFunctions()->supportFp16arith;
            break;
        }
        case STATUS_SUPPORT_DOT_PRODUCT: {
            return MNNGetCoreFunctions()->supportSDot;
            break;
        }
        default: {
            MNN_ERROR("unsupported interface");
            break;
        }
    }

    return 0;
}

void CPURuntime::onGabageCollect(int level) {
    mStaticAllocator->release(false);
}


void CPURuntime::onConcurrencyBegin() const {
#ifdef MNN_USE_THREAD_POOL
    if (mTaskIndex >= 0 && mPower != BackendConfig::Power_High) {
        ThreadPool::active();
    }
#else
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(mThreadNumber);
#endif
#endif
}

void CPURuntime::onConcurrencyEnd() const {
#ifdef MNN_USE_THREAD_POOL
    if (mTaskIndex >= 0 && mPower != BackendConfig::Power_High) {
        ThreadPool::deactive();
    }
#endif
}

std::map<OpType, CPUBackend::Creator*>* CPUBackend::gCreator = nullptr;
void CPUBackend::initCreatorMap() {
    gCreator = new std::map<OpType, CPUBackend::Creator*>;
}

bool CPUBackend::addCreator(OpType t, Creator* c) {
    auto map = gCreator;
    if (map->find(t) != map->end()) {
        MNN_PRINT("Error: %d type has be added\n", t);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}

CPUBackend::CPUBackend(const CPURuntime* runtime, BackendConfig::PrecisionMode precision, BackendConfig::MemoryMode memory, MNNForwardType type, size_t flags) : Backend(type) {
#ifdef LOG_VERBOSE
    MNN_PRINT("cpu backend create\n");
#endif
    mMemory = memory;
    mRuntime = const_cast<CPURuntime*>(runtime);
    std::shared_ptr<BufferAllocator::Allocator> defaultAlloc(BufferAllocator::Allocator::createRecurse(runtime->mStaticAllocator.get()));
    if (mRuntime->getAllocatorType() == Runtime::Allocator_Defer) {
        mDynamicAllocator.reset(new DeferBufferAllocator(defaultAlloc));
    } else {
        mDynamicAllocator.reset(new EagerBufferAllocator(defaultAlloc));
    }
    mCurrentDynamicAllocator = mDynamicAllocator.get();
    mStaticAllocator = runtime->mStaticAllocator;
    mPrecisionMode = precision;
    mCoreFunctions = MNNGetCoreFunctions();
    mInt8CoreFunctions = MNNGetInt8CoreFunctions();
    mCacheGroup.resize(2);
    for (int i=0; i<mCacheGroup.size(); ++i) {
        mCacheGroup[i].reset(new CPUResizeCache);
    }
    mCache = mCacheGroup[0].get();
}

CPUBackend::~CPUBackend() {
    mCacheGroup.clear();
}

void CPUBackend::onExecuteBegin() const {
    mRuntime->onConcurrencyBegin();
}

void CPUBackend::onExecuteEnd() const {
    mRuntime->onConcurrencyEnd();
}

void CPUBackend::onResizeBegin() {
    mCurrentDynamicAllocator->reset();
}
bool CPUBackend::onSelectDynamicAllocator(int index, int maxIndex) {
    if (maxIndex > 2) {
        return false;
    }
    if (maxIndex == 2 && mDynamicAllocatorBackup.get() == nullptr) {
        if (mRuntime->getAllocatorType() == Runtime::Allocator_Defer) {
            mDynamicAllocatorBackup.reset(new DeferBufferAllocator(BufferAllocator::Allocator::createRecurse(mStaticAllocator.get())));
        } else {
            mDynamicAllocatorBackup.reset(new EagerBufferAllocator(BufferAllocator::Allocator::createRecurse(mStaticAllocator.get())));
        }
    }
    if (1 == index) {
        mCurrentDynamicAllocator = mDynamicAllocatorBackup.get();
    } else {
        mCurrentDynamicAllocator = mDynamicAllocator.get();
    }
    mCache = mCacheGroup[index].get();
    return true;
}

ErrorCode CPUBackend::onResizeEnd() {
    getCache()->release();
    return mCurrentDynamicAllocator->compute();
}

Backend::MemObj* CPUBackend::allocBuffer(size_t size, Tensor* dest, StorageType storageType) {
    auto originMem = TensorUtils::getDescribeOrigin(dest)->mem.get();
    if (nullptr != originMem) {
        if (static_cast<CPUMemObj*>(originMem)->getSize() >= size) {
            return originMem;
        } else {
            TensorUtils::getDescribeOrigin(dest)->mem = nullptr;
        }
    }
    // MNN_PRINT("Acquire size = %d\n", size);
    if (size <= 0) {
        MNN_PRINT("Acquire buffer size = %lu\n", size);
        MNN_ASSERT(false);
        return nullptr;
    }
    // if (size > LARGE_MEMORY) {
    //     MNN_PRINT("Size larger than 500 M :%d\n", size);
    // }
    auto& buffer = dest->buffer();
    auto des = TensorUtils::getDescribe(dest);
    MemChunk chunk;
    switch (storageType) {
        case STATIC: {
            chunk = mStaticAllocator->alloc(size, false);
            break;
        }
        case DYNAMIC: {
            chunk = mCurrentDynamicAllocator->alloc(size, false);
            break;
        }
        case DYNAMIC_SEPERATE: {
            chunk = mCurrentDynamicAllocator->alloc(size, true);
            break;
        }
        default:
            MNN_ASSERT(false);
            break;
    }

    if (chunk.invalid()) {
        MNN_ERROR("Alloc buffer error for cpu backend\n");
        return nullptr;
    }

    Backend::MemObj* res = nullptr;

    if (storageType == STATIC) {
        res = new CPUMemObj(mStaticAllocator.get(), chunk, size);
    } else {
        res = new CPUMemObj(mCurrentDynamicAllocator, chunk, size);
        chunk.attach(dest);
    }
    if (chunk.ptr()) {
        buffer.host = chunk.ptr();
    }
    des->extra.offset = 0;
    return res;
}

Backend::MemObj* CPUBackend::onAcquire(const MNN::Tensor* nativeTensorConst, StorageType storageType) {
    if (nativeTensorConst == nullptr) {
        return nullptr;
    }
    //FUNC_PRINT_ALL(nativeTensorConst, p);
    auto nativeTensor = (Tensor*)nativeTensorConst;
    auto size = getTensorSize(nativeTensor, true);
    return allocBuffer(size, nativeTensor, storageType);
}

static OpType _getRealOpType(OpType opType) {
    switch (opType) {
        case OpType_Convolution:
            return OpType_ConvInt8;
        case OpType_ConvolutionDepthwise:
            return OpType_DepthwiseConvInt8;
        case OpType_Pooling:
            return OpType_PoolInt8;
        
        // case OpType_Eltwise:
        //     // TODO: just support EltwiseAdd
        //     return OpType_EltwiseInt8;
        default:
            return opType;
    }
}
void* CPUBackend::onMapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* srcTensor) {
    if (getBytes(this, srcTensor) != srcTensor->getType().bytes()) {
        return nullptr;
    }
    if (OpCommonUtils:: convertDimType(TensorUtils::getDescribe(srcTensor)->dimensionFormat) != dtype) {
        return nullptr;
    }
    return srcTensor->host<void>();
}

bool CPUBackend::onUnmapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* dstTensor, void* mapPtr) {
    if (getBytes(this, dstTensor) != dstTensor->getType().bytes()) {
        return false;
    }
    if (OpCommonUtils:: convertDimType(TensorUtils::getDescribe(dstTensor)->dimensionFormat) != dtype) {
        return false;
    }
    return true;
}

size_t CPUBackend::getTensorSize(const Tensor* tensor, bool multiBytes) const {
    auto core = mCoreFunctions;
    size_t dataSize = 1;
    auto des = TensorUtils::getDescribe(tensor);
    for (int i = 0; i < tensor->dimensions(); i++) {
        size_t currentDimSize = tensor->length(i);
        if (des->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && 1 == i) {
            currentDimSize = UP_DIV(currentDimSize, core->pack) * core->pack;
        }
        dataSize *= currentDimSize;
    }
    if (multiBytes) {
        size_t bytes = tensor->getType().bytes();
        if (TensorUtils::getDescribe(tensor)->quantAttr != nullptr) {
            if (TensorUtils::getDescribe(tensor)->type == DataType_DT_FLOAT) {
                bytes = 4;
            } else {
                bytes = 1;
            }
        }
        return dataSize * bytes;
    }
    return dataSize;
}

int CPUBackend::getBytes(const Backend* backend, const Tensor* output) {
    auto bytes = output->getType().bytes();
    auto core = static_cast<const CPUBackend*>(backend)->functions();
    auto quant = TensorUtils::getDescribe(output)->quantAttr.get();
    if (output->getType().code == halide_type_float) {
        bytes = core->bytes;
    }
    if (nullptr != quant && TensorUtils::getDescribe(output)->type == DataType_DT_INT8) {
        bytes = 1;
    }
    return bytes;
}

DataType CPUBackend::getDataType(const Tensor* tensor) {
    auto des = TensorUtils::getDescribe(tensor);
    if (nullptr == des->quantAttr.get()) {
        return DataType_DT_FLOAT;
    }
    return des->type;
}

/// get execution
Execution* CPUBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) {
    /**
     BatchNorm it will be converted to scale
     for model convert, don't print error log
     */
    if (op->type() == OpType_BatchNorm) {
        return nullptr;
    }
    auto opType = op->type();
    if (outputs.size() > 0) {
        if (TensorUtils::getDescribe(outputs[0])->quantAttr != nullptr && TensorUtils::getDescribe(outputs[0])->type == DataType_DT_INT8) {
            opType = _getRealOpType(opType);
        }
    }

    // TODO: rm this convert when merge diff datatyoe of op
    auto map  = gCreator;
    auto iter = map->find(opType);
    if (iter == map->end()) {
        MNN_PRINT("Don't support type [%s], %s\n", MNN::EnumNameOpType(op->type()), op->name()->c_str());
        return nullptr;
    }
    Execution* exe = nullptr;
    bool needCast = false;
    if (exe == nullptr) {
        exe = iter->second->onCreate(inputs, outputs, op, this);
    }
    return exe;
}
const Runtime* CPUBackend::getRuntime() {
    return mRuntime;
}

bool CPUBackend::onClearBuffer() {
    mCache->reset();
    mCurrentDynamicAllocator->release(true);
    return true;
}

std::pair<int, int> CPUBackend::multiThreadDivide(int size) const {
    int sizeDivide = size / threadNumber();
    sizeDivide = UP_DIV(sizeDivide, mCoreFunctions->pack) * mCoreFunctions->pack;
    int scheduleNumber = 1;
    if (sizeDivide > 0) {
        scheduleNumber = UP_DIV(size, sizeDivide);
    }
    return std::make_pair(sizeDivide, scheduleNumber);
}
void CPUBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    auto& srcBuffer = srcTensor->buffer();
    auto& dstBuffer = dstTensor->buffer();

    MNN_ASSERT(srcBuffer.dimensions == dstBuffer.dimensions);
    if (srcTensor->getDimensionType() == dstTensor->getDimensionType()) {
        for (int i = 0; i < srcBuffer.dimensions; ++i) {
            MNN_ASSERT(srcBuffer.dim[i].extent <= dstBuffer.dim[i].extent);
        }
    }
    if (nullptr == srcBuffer.host || nullptr == dstBuffer.host) {
        return;
    }
    std::unique_ptr<Tensor> wrapTensor;
    if (getDataType(srcTensor) != getDataType(dstTensor)) {
        auto dimType =  OpCommonUtils::convertDimType(TensorUtils::getDescribe(srcTensor)->dimensionFormat);
        auto convertType = CPUCastCreator::FlOAT_TO_INT8;
        if (getDataType(srcTensor) == DataType_DT_INT8) {
            convertType = CPUCastCreator::INT8_TO_FlOAT;
        }
        wrapTensor.reset(Tensor::createDevice(srcTensor->shape(), dstTensor->getType(), dimType));
        auto dstType = getDataType(dstTensor);
        if (dstType != DataType_DT_FLOAT) {
            wrapTensor->setType(dstType);
        }
        wrapTensor->buffer().host = (uint8_t*)MNNMemoryAllocAlign(getTensorSize(wrapTensor.get()) * wrapTensor->getType().bytes(), MNN_MEMORY_ALIGN_DEFAULT);

#ifdef LOG_VERBOSE
        MNN_PRINT("CPU backend copy tensor ptr:%p -> ptr:%p hostPtr:%p -> %p, format %d -> %d, dims: [",
        srcTensor, dstTensor, srcTensor->host<void>(), dstTensor->host<void>(), TensorUtils::getDescribe(srcTensor)->dimensionFormat, TensorUtils::getDescribe(dstTensor)->dimensionFormat);
        for (int i=0; i<srcTensor->dimensions(); ++i) {
            MNN_PRINT("%d ", srcTensor->length(i));
        }
        MNN_PRINT("]\n");
#endif

        TensorUtils::getDescribe(wrapTensor.get())->memoryType = Tensor::InsideDescribe::MEMORY_HOST;
        auto code = CPUCastCreator::cast(srcTensor, wrapTensor.get(), this, convertType);
        if (NO_ERROR != code) {
            MNN_ERROR("Error in CPUBackend::onCopyBuffer:cast\n");
        }
        srcTensor = wrapTensor.get();
    } else if (srcTensor->getType() != dstTensor->getType()) {
        MNN_ERROR("Input type not match session's tensor\n");
        return;
    }
    auto code = CPUTensorConverter::convert(srcTensor, dstTensor);
    if (NO_ERROR != code) {
        MNN_ERROR("Error in CPUBackend::onCopyBuffer:convert\n");
    }
}

class CPURuntimeCreator : public RuntimeCreator {
public:
    virtual Runtime* onCreate(const Backend::Info& info) const override {
        return new CPURuntime(info);
    }
};


#ifdef MNN_SUPPORT_BF16
extern void registerBF16Backend();
#endif
#ifdef ENABLE_ARMV82
extern void registerArm82RuntimeCreator();
#endif
void registerCPURuntimeCreator() {
    MNNCoreFunctionInit();
    CPUBackend::initCreatorMap();
    registerCPUOps();
#ifdef MNN_SUPPORT_BF16
    registerBF16Backend();
#endif
#ifdef MNN_USE_ARMV82
    registerArm82RuntimeCreator();
#endif
    // TODO: Merge _initCoreFunction MNNFunctionInit and cpuinfo_arm_init
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_CPU, new CPURuntimeCreator);
};
} // namespace MNN
