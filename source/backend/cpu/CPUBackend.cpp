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
#include "core/BufferAllocator.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/TensorUtils.hpp"
#include "backend/cpu/ThreadPool.hpp"
#include "shape/SizeComputer.hpp"
#include "compute/CommonOptFunction.h"
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP
#include "backend/cpu/CPURuntime.hpp"
#if defined(__aarch64__) && ENABLE_ARMV82
#include "backend/arm82/Arm82Backend.hpp"
#endif
#define MAX_THREAD_NUMBER 32
#define LARGE_MEMORY 1024 * 1024 * 100

//#define MNN_DUMP_MEMORY_USAGE
#define MNN_CPU_CHECK_NAN 1
namespace MNN {
void registerCPUOps();
#if defined(__aarch64__) && ENABLE_ARMV82
struct cpuinfo_arm_isa gCPUInfo;
#endif

CPURuntime::CPURuntime(const Backend::Info& info) {
    mDynamicAllocator.reset(new BufferAllocator);
    mStaticAllocator.reset(new BufferAllocator);
    mThreadNumber = info.numThread;
    mThreadNumber = std::max(1, mThreadNumber);
    mThreadNumber = std::min(mThreadNumber, MAX_THREAD_NUMBER);
    mPower   = BackendConfig::Power_Normal;
    mMemory  = BackendConfig::Memory_Normal;
    mPrecision = BackendConfig::Precision_Normal;
    mFlags = 0;
    mFlops = MNNGetCPUFlops(mThreadNumber);
#if defined(__aarch64__) && ENABLE_ARMV82
    mIsSupportDot = gCPUInfo.dot;
    mIsSupportFp16arith = gCPUInfo.fp16arith;
#endif
    if (info.user != nullptr) {
        mPrecision = info.user->precision;
        mPower = info.user->power;
        mMemory = info.user->memory;
        mFlags = info.user->flags;
    }
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
    auto dynamicMemoryInMB = mDynamicAllocator->totalSize() / 1024.0f / 1024.0f;
    auto staticMemoryInMB = mStaticAllocator->totalSize() / 1024.0f / 1024.0f;
    return dynamicMemoryInMB + staticMemoryInMB;
}
Backend* CPURuntime::onCreate() const{
#if defined(__aarch64__) && ENABLE_ARMV82
    if (mIsSupportFp16arith && mPrecision == BackendConfig::Precision_Low) {
        return new Arm82Backend(this);
    }
#endif
    return new CPUBackend(this);
}
void CPURuntime::onGabageCollect(int level) {
    mStaticAllocator->release(false);
    if (level > 50) {
        mDynamicAllocator->release(false);
    }
}
std::map<OpType, CPUBackend::Creator*>* CPUBackend::gCreator = nullptr;

void CPUBackend::initCreatorMap() {
    gCreator = new std::map<OpType, CPUBackend::Creator*>;
}

std::map<OpType, CPUBackend::Creator*>* CPUBackend::getCreatorMap() {
    return gCreator;
}

bool CPUBackend::addCreator(OpType t, Creator* c) {
    auto map = getCreatorMap();
    if (map->find(t) != map->end()) {
        MNN_PRINT("Error: %d type has be added\n", t);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}

CPUBackend::CPUBackend(const CPURuntime* runtime, MNNForwardType type) : Backend(type) {
    mRuntime = runtime;
    mCheckNAN = runtime->mFlags == MNN_CPU_CHECK_NAN;
    mDynamicAllocator = runtime->mDynamicAllocator;
    mStaticAllocator = runtime->mStaticAllocator;
}
bool CPUBackend::supportDot() const {
    return mRuntime->mIsSupportDot;
}

CPUBackend::~CPUBackend() {
    for (auto p : mDynamic) {
        mDynamicAllocator->free(p);
    }
}

void CPUBackend::onExecuteBegin() const {
#ifdef MNN_USE_THREAD_POOL
    if (mRuntime->mTaskIndex >= 0 && mRuntime->mPower != BackendConfig::Power_High) {
        ThreadPool::active();
    }
#else
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(threadNumber());
#endif
#endif
}
void CPUBackend::onExecuteEnd() const {
#ifdef MNN_USE_THREAD_POOL
    if (mRuntime->mTaskIndex >= 0 && mRuntime->mPower != BackendConfig::Power_High) {
        ThreadPool::deactive();
    }
#endif
}

bool CPUBackend::allocBuffer(int size, halide_buffer_t& buffer, StorageType storageType) {
    // MNN_PRINT("Acquire size = %d\n", size);
    if (size <= 0) {
        MNN_ASSERT(false);
        return false;
    }
    if (size > LARGE_MEMORY) {
        MNN_PRINT("Size larger the 100 M :%d\n", size);
    }
    switch (storageType) {
        case STATIC: {
#ifdef MNN_DUMP_MEMORY_USAGE
            buffer.host = (uint8_t*)malloc(size);
#else
            buffer.host = (uint8_t*)(mStaticAllocator->alloc(size, false));
#endif
            break;
        }
        case DYNAMIC: {
            buffer.host = (uint8_t*)(mDynamicAllocator->alloc(size, false));
            break;
        }
        case DYNAMIC_SEPERATE: {
            buffer.host = (uint8_t*)(mDynamicAllocator->alloc(size, true));
            break;
        }
        default:
            MNN_ASSERT(false);
            break;
    }
    if (nullptr == buffer.host) {
        MNN_ERROR("Alloc buffer error for cpu backend\n");
        return false;
    }
    if (STATIC == storageType) {
        // Do nothing
    } else {
        mDynamic.insert(buffer.host);
    }
    if (buffer.type.code == halide_type_handle) {
        ::memset(buffer.host, 0, size);
    }
    return true;
}

bool CPUBackend::onAcquireBuffer(const MNN::Tensor* nativeTensorConst, StorageType storageType) {
    if (nativeTensorConst == nullptr) {
        return false;
    }
    //FUNC_PRINT_ALL(nativeTensorConst, p);
    auto nativeTensor = (Tensor*)nativeTensorConst;
    auto& buffer      = nativeTensor->buffer();

    auto size = nativeTensor->size();
    return allocBuffer(size, buffer, storageType);
}

bool CPUBackend::onReleaseBuffer(const MNN::Tensor* nativeTensor, StorageType storageType) {
    if (nativeTensor == nullptr) {
        return false;
    }
    if (nullptr == nativeTensor->buffer().host) {
        return false;
    }
    if (STATIC == storageType) {
#ifdef MNN_DUMP_MEMORY_USAGE
        free(nativeTensor->buffer().host);
#else
        mStaticAllocator->free(nativeTensor->buffer().host);
#endif
        return true;
    }
    if (DYNAMIC_SEPERATE == storageType) {
        return true;
    }
    mDynamic.erase(nativeTensor->buffer().host);
    mDynamicAllocator->free(nativeTensor->buffer().host);
    return true;
}

std::pair<float, bool> CPUBackend::onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const MNN::Op* op) {
    auto map  = getCreatorMap();
    auto iter = map->find(op->type());
    if (iter == map->end()) {
        MNN_PRINT("Don't support type %s, %s\n", MNN::EnumNameOpType(op->type()), op->name()->c_str());
        return std::make_pair(0.0f, false);
    }
#ifndef MNN_BUILD_MINI
    auto computeFlops = SizeComputer::computeFlops(op, inputs, outputs);
    return std::make_pair(computeFlops / mRuntime->mFlops * 1000.0f, true);
#else
    return std::make_pair(0.0f, false);
#endif
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
    auto map  = getCreatorMap();
    auto iter = map->find(op->type());
    if (iter == map->end()) {
        MNN_PRINT("Don't support type [%s], %s\n", MNN::EnumNameOpType(op->type()), op->name()->c_str());
        return nullptr;
    }
    auto exe = iter->second->onCreate(inputs, outputs, op, this);
    if (nullptr == exe) {
        MNN_PRINT("The Creator Don't support type [%s], %s\n", MNN::EnumNameOpType(op->type()), op->name()->c_str());
        return nullptr;
    }
    if (mCheckNAN) {
        class CheckNANExecution : public Execution {
        public:
            CheckNANExecution(Execution* exe) : Execution(exe->backend()) {
                mExecution.reset(exe);
                mValid = exe->valid();
            }
            virtual ~CheckNANExecution() {
                // Do nothing
            }
            virtual ErrorCode onResize(const std::vector<Tensor*>& inputs,
                                       const std::vector<Tensor*>& outputs) override {
                return mExecution->onResize(inputs, outputs);
            }

            virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs,
                                        const std::vector<Tensor*>& outputs) override {
                for (auto tensor : inputs) {
                    if (halide_type_float != tensor->getType().code) {
                        return NO_ERROR;
                    }
                    auto size = tensor->elementSize();
                    auto ptr  = tensor->host<float>();
                    for (int i = 0; i < size; ++i) {
                        auto value = ptr[i];
                        if (std::isnan(value) || std::isinf(value)) {
                            return INVALID_VALUE;
                        }
                    }
                }
                auto code = mExecution->onExecute(inputs, outputs);
                if (NO_ERROR != code) {
                    return code;
                }
                for (auto tensor : outputs) {
                    if (halide_type_float != tensor->getType().code) {
                        return NO_ERROR;
                    }
                    auto size = tensor->elementSize();
                    auto ptr  = tensor->host<float>();
                    for (int i = 0; i < size; ++i) {
                        auto value = ptr[i];
                        if (std::isnan(value) || std::isinf(value)) {
                            return INVALID_VALUE;
                        }
                    }
                }
                return NO_ERROR;
            }

        private:
            std::unique_ptr<Execution> mExecution;
        };
        return new CheckNANExecution(exe);
    }
    return exe;
}

bool CPUBackend::onClearBuffer() {
    for (auto p : mDynamic) {
        mDynamicAllocator->free(p);
    }
    mDynamic.clear();
    return true;
}

std::pair<int, int> CPUBackend::multiThreadDivide(int size) const {
    int sizeDivide = size / threadNumber();
    sizeDivide = UP_DIV(sizeDivide, 4) * 4;
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
    MNN_ASSERT(srcBuffer.type == dstBuffer.type);
    if (srcTensor->getDimensionType() == dstTensor->getDimensionType()) {
        for (int i = 0; i < srcBuffer.dimensions; ++i) {
            MNN_ASSERT(srcBuffer.dim[i].extent <= dstBuffer.dim[i].extent);
        }
    }
    if (nullptr == srcBuffer.host || nullptr == dstBuffer.host) {
        return;
    }

    auto code = CPUTensorConverter::convert(srcTensor, dstTensor);
    if (NO_ERROR != code) {
        MNN_ERROR("Error in CPUBackend::onCopyBuffer\n");
    }
}

class CPURuntimeCreator : public RuntimeCreator {
public:
    virtual Runtime* onCreate(const Backend::Info& info) const override {
        return new CPURuntime(info);
    }
};


void registerCPURuntimeCreator() {
    CPUBackend::initCreatorMap();
    registerCPUOps();
    MNNFunctionInit();
#if defined(__aarch64__) && ENABLE_ARMV82
    cpuinfo_arm_init(&gCPUInfo);
#endif
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_CPU, new CPURuntimeCreator);
};
} // namespace MNN
