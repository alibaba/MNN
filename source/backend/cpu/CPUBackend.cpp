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
#include "core/MNNFileUtils.h"
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
#include "bf16/BF16Functions.hpp"
#endif

#ifdef MNN_USE_SSE
#include "x86_x64/AVX2Backend.hpp"
#endif

#define MNN_CPU_MAX_BUFFER_INDEX 2
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
void CPUBackend::computeDivideSizes(int size, int* dst, float avgDiv) const {
    if (mGroupWithComputeRate.size() <= 1 || (avgDiv > 0 && avgDiv < mComputeI)) {
        // Avg divide
        int length = UP_DIV(size, mThreadNumber);
        int cur = length;
        for (int i=0; i<mThreadNumber; ++i) {
            dst[i] = cur;
            cur = cur + length;
            cur = ALIMIN(cur, size);
        }
        return;
    }

    int cur = 0;
    int curPos = 0;
    for (auto& group : mGroupWithComputeRate) {
        int currentGroupTotal = (int)(ceilf((float)size*group.first));
        int length = UP_DIV(currentGroupTotal, group.second);
        for (int i=0; i<group.second; ++i) {
            cur = cur + length;
            cur = ALIMIN(cur, size);
            dst[curPos+i] = cur;
        }
        curPos += group.second;
    }
}

void CPURuntime::_bindCPUCore() const {
    if (mPower == BackendConfig::Power_Normal) {
        return;
    }
    auto tid = MNNGetCurrentPid();
    if (tid == mCurrentTID) {
        return;
    }
    mCurrentTID = tid;
    // Bind CPU Core
    auto cpuInfo = MNNGetCPUInfo();
    if (cpuInfo->groups.size() == 0) {
        return;
    }
    std::vector<std::pair<const int*, int>> lockCPUIndexes(mThreadNumber);
    switch (mPower) {
        case BackendConfig::Power_Low:
            for (int v=0; v<mThreadNumber; ++v) {
                lockCPUIndexes[v] = std::make_pair(cpuInfo->groups[0].ids.data(), cpuInfo->groups[0].ids.size());
            }
            break;
        case BackendConfig::Power_High:
        {
            int selectCPUSize = 0;
            int groupIndex = cpuInfo->groups.size() - 1;
            while (selectCPUSize < mThreadNumber && groupIndex >= 0) {
                auto& group = cpuInfo->groups[groupIndex];
                int size = ALIMIN(group.ids.size(), mThreadNumber - selectCPUSize);
                for (int v=0; v<size; ++v) {
                    lockCPUIndexes[v + selectCPUSize] = std::make_pair(group.ids.data(), group.ids.size());
                }
                groupIndex--;
                selectCPUSize += group.ids.size();
            }
        }
            break;
        default:
            break;
    }
        // Set CPU Affinity
#ifdef _OPENMP
    auto threadsNumber = mThreadNumber;
    std::vector<int> result(threadsNumber, 0);
#pragma omp parallel for
    for (int i = 0; i < threadsNumber; ++i) {
        result[i] = MNNSetSchedAffinity(lockCPUIndexes[i].first, lockCPUIndexes[i].second);
    }
#endif
#ifdef MNN_USE_THREAD_POOL
    ThreadPool::active(mThreadNumber);
    ThreadPool::enqueue(std::make_pair([&](int i) {
        MNNSetSchedAffinity(lockCPUIndexes[i].first, lockCPUIndexes[i].second);
        return 0;
    }, mThreadNumber), mTaskIndex, mThreadNumber);
    ThreadPool::deactive(mThreadNumber);
#endif
}

void CPURuntime::_resetThreadPool() {
    mThreadNumber = std::max(1, mThreadNumber);
    mThreadNumber = std::min(mThreadNumber, MAX_THREAD_NUMBER);
#ifdef MNN_USE_THREAD_POOL
    ThreadPool::releaseWorkIndex(mTaskIndex);
    auto cpuInfo = MNNGetCPUInfo();
    if (mThreadNumber > 1) {
        int systemThreadNumber = (int)cpuInfo->cpuNumber;
        if (systemThreadNumber == 0) {
            systemThreadNumber = mThreadNumber;
        }
        mThreadNumber = ALIMIN(ThreadPool::init(systemThreadNumber), mThreadNumber);
    }
    if (mThreadNumber > 1) {
        mTaskIndex = ThreadPool::acquireWorkIndex();
        if (-1 == mTaskIndex) {
            MNN_ERROR("The ThreadPool has been used to MNN_THREAD_POOL_MAX_TASKS, can't use thread pool\n");
            mThreadNumber = 1;
        }
    } else {
        mTaskIndex = -1;
    }
#endif
    // Reset tid to rebind cpu if necessary
    mCurrentTID = 0;
}
void CPURuntime::onReset(int numberThread, const BackendConfig* config, bool full) {
    if (config != nullptr) {
        mPower = config->power;
        if (full) {
            mPrecision = config->precision;
            mMemory = config->memory;
            mFlags = config->flags;
        }
    }
    mThreadNumber = numberThread;
    _resetThreadPool();
}

CPURuntime::CPURuntime(const Backend::Info& info) {
    auto rawAlloc = BufferAllocator::Allocator::createDefault();
    mStaticAllocator.reset(new EagerBufferAllocator(rawAlloc));
    mDynamic.resize(MNN_CPU_MAX_BUFFER_INDEX);
    for (auto& buf : mDynamic) {
        buf.root = rawAlloc;
    }
    mThreadNumber = info.numThread;
    mPower   = BackendConfig::Power_Normal;
    mMemory  = BackendConfig::Memory_Normal;
    mPrecision = BackendConfig::Precision_Normal;
    if (info.user != nullptr) {
        mPrecision = info.user->precision;
        mPower = info.user->power;
        mMemory = info.user->memory;
        mFlags = info.user->flags;
    }
    _resetThreadPool();
#ifdef LOG_VERBOSE
    MNN_PRINT("create CPURuntime:%p\n", this);
#endif
}

CPURuntime:: ~ CPURuntime() {
#ifdef MNN_USE_THREAD_POOL
    ThreadPool::releaseWorkIndex(mTaskIndex);
#endif
}
float CPURuntime::onGetMemoryInMB() {
    auto staticMemoryInMB = mStaticAllocator->totalSize() / 1024.0f / 1024.0f;
    float dynamicMemoryInMB = 0.0f;
    for (auto& buf : mDynamic) {
        dynamicMemoryInMB += buf.currentSize / 1024.0f / 1024.0f;
    }
    return staticMemoryInMB + dynamicMemoryInMB;
}
bool CPURuntime::onCheckInfo(Backend::Info& info) const {
    info.numThread = mThreadNumber;
    return true;
}
SingleBufferWithAllocator* CPURuntime::buffer(int index) const {
    if (mDynamicMmap.empty()) {
        return mDynamic.data() + index;
    }
    return mDynamicMmap.data() + index;
}

Backend* CPURuntime::onCreate(const BackendConfig* config, Backend* origin) const {
    if (hint().midMemoryPath.size() > 0) {
        if (mDynamicMmap.empty()) {
            // Only support set featuremap dir once
            mDynamicMmap.resize(2);
            auto mmapMem = BufferAllocator::Allocator::createMmap(hint().midMemoryPath.c_str(), "", "dynamic");
            for (auto& buf : mDynamicMmap) {
                buf.root = mmapMem;
            }
        }
    }
    if (hint().weightMemoryPath.size() > 0) {
        // forward_type, precision_type, memory_type, power_type
        std::string prefix = "0_0_0_0_";
        prefix[2] += mPrecision;
        prefix[4] += mMemory;
        prefix[6] += mPower;
        // prefix += hint().modelUUID + "_";
        bool autoRemove = true;
        if (hint().useCachedMmap) {
            autoRemove = false;
            std::string fileName = MNNFilePathConcat(hint().weightMemoryPath, prefix + "sync.static");
            const_cast<RuntimeHint&>(hint()).useCachedMmap += MNNFileExist(fileName.c_str());
        }
        if (nullptr == mStaticAllocatorCache.get()) {
            // Only support set weightmap dir once
            mStaticAllocatorCache = mStaticAllocator;
            auto mmapMem = BufferAllocator::Allocator::createMmap(hint().weightMemoryPath.c_str(), prefix.c_str(), "static", autoRemove);
            size_t mmapSize = static_cast<size_t>(hint().mmapFileSize) * 1024 * 1024;
            mStaticAllocator.reset(new EagerBufferAllocator(mmapMem, 32, mmapSize));
        }
    }
    auto precision = mPrecision;
    auto memory = mMemory;
    size_t flags = mFlags;
    if (nullptr != origin) {
        auto cpuBn = static_cast<CPUBackend*>(origin);
        mSharedDmaInfo = cpuBn->mDmaInfo;
    }
    if (nullptr != config) {
        precision = config->precision;
        flags = config->flags;
        memory = config->memory;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("cpu backend was created by runtime:%p\n", this);
#endif
    CPUBackend* res = nullptr;
    do {
#ifdef MNN_USE_ARMV82
        auto core = MNNGetCoreFunctions();
        if (core->supportFp16arith && precision == BackendConfig::Precision_Low) {
            res = new Arm82Backend(this, memory);
            break;
        }
#endif
#ifdef MNN_SUPPORT_BF16
        if (precision == BackendConfig::Precision_Low_BF16 && BF16Functions::get()) {
            res = new CPUBackend(this, precision, memory, MNN_FORWARD_CPU_EXTENSION, 0);
            res->mCoreFunctions = BF16Functions::get();
            break;
        }
#endif
        if (flags == MNN_CPU_USE_DEFAULT_BACKEND) {
            res = new CPUBackend(this, precision, memory, MNN_FORWARD_CPU, 0);
            break;
        }
#ifdef MNN_USE_SSE
        if (AVX2Backend::isValid()) {
            res = new AVX2Backend(this, memory, flags);
            break;
        }
#endif
        res = new CPUBackend(this, precision, memory, MNN_FORWARD_CPU, flags);
    } while (false);
    mSharedDmaInfo = nullptr;
    return res;
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
    if (level >= 100) {
        for (auto& buf : mDynamic) {
            buf.release();
        }
    }
}


void CPURuntime::onConcurrencyBegin() const {
#ifdef MNN_USE_THREAD_POOL
    if (mTaskIndex >= 0) {
        ThreadPool::active(mThreadNumber);
        mThreadOpen = true;
    }
#else
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(mThreadNumber);
#endif
#endif
    _bindCPUCore();
}

void CPURuntime::onConcurrencyEnd() const {
#ifdef MNN_USE_THREAD_POOL
    if (mTaskIndex >= 0) {
        ThreadPool::deactive(mThreadNumber);
        mThreadOpen = false;
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
BufferAllocator* CPURuntime::createDynamicBufferAlloctor(int index) const {
    if (hint().memoryAllocatorType == Runtime::Allocator_Defer) {
        return new DeferBufferAllocator(buffer(index));
    }
    if (nullptr != mStaticAllocatorCache.get()) {
        return new EagerBufferAllocator(BufferAllocator::Allocator::createRecurse(mStaticAllocatorCache.get()));
    }
    return new EagerBufferAllocator(BufferAllocator::Allocator::createRecurse(mStaticAllocator.get()));
}
CPUBackend::CPUBackend(const CPURuntime* runtime, BackendConfig::PrecisionMode precision, BackendConfig::MemoryMode memory, MNNForwardType type, size_t flags) : Backend(type) {
#ifdef LOG_VERBOSE
    MNN_PRINT("cpu backend create\n");
#endif
    mMemory = memory;
    mRuntime = const_cast<CPURuntime*>(runtime);
    mThreadNumber = mRuntime->mThreadNumber;
    // Compute Group Rate
    do {
        if (mThreadNumber <= 1 || mRuntime->mPower == BackendConfig::Power_Low) {
            break;
        }
        auto rate = mRuntime->hint().cpuDecreaseRate;
        if (rate >= 100 || rate <= 0) {
            break;
        }
        auto cpuInfo = MNNGetCPUInfo();
        if (cpuInfo->groups.size() < 2) {
            break;
        }
        if (cpuInfo->i8mm) {
            mComputeI = 28.f;
        } else if (cpuInfo->dot) {
            mComputeI = 14.f;
        } else {
            mComputeI = 7.f;
        }
        mGroupWithComputeRate.clear();
        float decreaseRate = (float)(rate) / 100.0f;
        int validCpuSize = (int)(cpuInfo->groups[cpuInfo->groups.size()-1].ids.size());
        int groupIndex = (int)cpuInfo->groups.size()-2;
        validCpuSize = ALIMIN(validCpuSize, mThreadNumber);
        float totalComputeRate = 1.0f * validCpuSize;
        mGroupWithComputeRate.emplace_back(std::make_pair(totalComputeRate, validCpuSize));
        float currentRate = 1.0f;
        while (validCpuSize < mThreadNumber && groupIndex >= 0) {
            auto& group = cpuInfo->groups[groupIndex];
            int selectSize = ALIMIN(mThreadNumber - validCpuSize, (int)group.ids.size());
            validCpuSize += group.ids.size();
            currentRate *= decreaseRate;
            totalComputeRate += currentRate * selectSize;
            mGroupWithComputeRate.emplace_back(std::make_pair(currentRate * selectSize, selectSize));
        }
        for (auto& g : mGroupWithComputeRate) {
            g.first = g.first / totalComputeRate;
        }
    } while (false);
    auto dynamicAlloc = mRuntime->mSharedDmaInfo;
    if (nullptr == dynamicAlloc.get()) {
        mDmaInfo.reset(new CPURuntime::DynamicAllocator);
        mDmaInfo->mDynamicAllocator.reset(mRuntime->createDynamicBufferAlloctor(0));
        mDmaInfo->mCurrentDynamicAllocator = mDmaInfo->mDynamicAllocator.get();
    } else {
        mDmaInfo = dynamicAlloc;
    }
    mStaticAllocator = runtime->mStaticAllocator;
    mPrecisionMode = precision;
    mCoreFunctions = MNNGetCoreFunctions();
    mInt8CoreFunctions = MNNGetInt8CoreFunctions();
    mCacheGroup.resize(MNN_CPU_MAX_BUFFER_INDEX);
    for (int i=0; i<mCacheGroup.size(); ++i) {
        mCacheGroup[i].reset(new CPUResizeCache);
    }
    mCache = mCacheGroup[0].get();
}

CPUBackend::~CPUBackend() {
    mCacheGroup.clear();
}
void CPUBackend::_resetDynamicMemory() const {
    mRuntime->pCurrentStatus = mDmaInfo->mDynamicAllocator->apply();
    if (NO_ERROR != mRuntime->pCurrentStatus) {
        return;
    }
    if (nullptr != mDmaInfo->mDynamicAllocatorBackup.get()) {
        mRuntime->pCurrentStatus  = mDmaInfo->mDynamicAllocatorBackup->apply();
    }
}

void CPUBackend::onExecuteBegin() const {
    _resetDynamicMemory();
    mRuntime->onConcurrencyBegin();
}

void CPUBackend::onExecuteEnd() const {
    mRuntime->onConcurrencyEnd();
}

void CPUBackend::onResizeBegin() {
    mDmaInfo->mCurrentDynamicAllocator->reset();
}
bool CPUBackend::onSelectDynamicAllocator(int index, int maxIndex) {
    if (maxIndex > 2) {
        return false;
    }
    if (maxIndex == 2 && mDmaInfo->mDynamicAllocatorBackup.get() == nullptr) {
        mDmaInfo->mDynamicAllocatorBackup.reset(mRuntime->createDynamicBufferAlloctor(1));
    }
    if (1 == index) {
        mDmaInfo->mCurrentDynamicAllocator = mDmaInfo->mDynamicAllocatorBackup.get();
    } else {
        mRuntime->buffer(0)->release();
        mDmaInfo->mCurrentDynamicAllocator = mDmaInfo->mDynamicAllocator.get();
    }
    mCache = mCacheGroup[index].get();
    return true;
}

ErrorCode CPUBackend::onResizeEnd() {
    getCache()->release();
    auto code = mDmaInfo->mCurrentDynamicAllocator->compute();
    if (NO_ERROR != code) {
        return code;
    }
    return NO_ERROR;
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
            chunk = mDmaInfo->mCurrentDynamicAllocator->alloc(size, false);
            break;
        }
        case DYNAMIC_SEPERATE: {
            chunk = mDmaInfo->mCurrentDynamicAllocator->alloc(size, true);
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
        res = new CPUMemObj(mDmaInfo->mCurrentDynamicAllocator, chunk, size);
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
    if (iter == map->end() ) {
        MNN_PRINT("Don't support type [%s]\n", MNN::EnumNameOpType(op->type()));
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
    if (nullptr != mRuntime->mStaticAllocatorCache.get()) {
        mStaticAllocator->sync();
        mStaticAllocator = mRuntime->mStaticAllocatorCache;
    }
    mCache->reset();
    mDmaInfo->mCurrentDynamicAllocator->release(true);
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
    _resetDynamicMemory();
    auto& srcBuffer = srcTensor->buffer();
    auto& dstBuffer = dstTensor->buffer();
    if (srcBuffer.dimensions != dstBuffer.dimensions ) {
        if (srcBuffer.dim[srcBuffer.dimensions - 1].extent != 1 && dstBuffer.dim[dstBuffer.dimensions - 1].extent != 1) {
            MNN_ERROR("srcBuffer dimension not equal to dstBuffer, can't copy buffer\n");
        }
    }
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
