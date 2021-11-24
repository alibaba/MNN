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

CPURuntime::CPURuntime(const Backend::Info& info) {
    mStaticAllocator.reset(new BufferAllocator(BufferAllocator::Allocator::createDefault()));
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
    auto staticMemoryInMB = mStaticAllocator->totalSize() / 1024.0f / 1024.0f;
    return staticMemoryInMB;
}
Backend* CPURuntime::onCreate(const BackendConfig* config) const {
    auto precision = mPrecision;
    size_t flags = mFlags;
    if (nullptr != config) {
        precision = config->precision;
        flags = config->flags;
    }
#ifdef MNN_USE_ARMV82
    auto core = MNNGetCoreFunctions();
    if (core->supportFp16arith && precision == BackendConfig::Precision_Low) {
        return new Arm82Backend(this);
    }
#endif
#ifdef MNN_SUPPORT_BF16
    if (precision == BackendConfig::Precision_Low && BF16Functions::get()) {
        return new BF16Backend(this);
    }
#endif
    if (flags == MNN_CPU_USE_DEFAULT_BACKEND) {
        return new CPUBackend(this, precision, MNN_FORWARD_CPU, 0);
    }
#ifdef MNN_USE_SSE
    if (AVX2Backend::isValid()) {
        return new AVX2Backend(this, flags);
    }
#endif
    return new CPUBackend(this, precision, MNN_FORWARD_CPU, flags);
}
void CPURuntime::onGabageCollect(int level) {
    mStaticAllocator->release(false);
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

CPUBackend::CPUBackend(const CPURuntime* runtime, BackendConfig::PrecisionMode precision, MNNForwardType type, size_t flags) : Backend(type) {
    mRuntime = runtime;
    mCheckNAN = flags == MNN_CPU_CHECK_NAN;
    std::shared_ptr<BufferAllocator::Allocator> defaultAlloc(BufferAllocator::Allocator::createRecurse(runtime->mStaticAllocator.get()));
    mDynamicAllocator.reset(new BufferAllocator(defaultAlloc));
    mStaticAllocator = runtime->mStaticAllocator;
    mPrecisionMode = precision;
    mCoreFunctions = MNNGetCoreFunctions();
    mInt8CoreFunctions = MNNGetInt8CoreFunctions();
}

CPUBackend::~CPUBackend() {
    // Do nothing
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

bool CPUBackend::allocBuffer(int size, Tensor* dest, StorageType storageType) {
    // MNN_PRINT("Acquire size = %d\n", size);
    if (size <= 0) {
        MNN_PRINT("Acquire buffer size = %d\n", size);
//        MNN_ASSERT(false);
        return false;
    }
    // if (size > LARGE_MEMORY) {
    //     MNN_PRINT("Size larger than 500 M :%d\n", size);
    // }
    auto& buffer = dest->buffer();
    auto des = TensorUtils::getDescribe(dest);
    std::pair<void*, int> points;
    switch (storageType) {
        case STATIC: {
            points = mStaticAllocator->alloc(size, false);
            break;
        }
        case DYNAMIC: {
            points = mDynamicAllocator->alloc(size, false);
            break;
        }
        case DYNAMIC_SEPERATE: {
            points = mDynamicAllocator->alloc(size, true);
            break;
        }
        default:
            MNN_ASSERT(false);
            break;
    }
    if (nullptr == points.first) {
        MNN_ERROR("Alloc buffer error for cpu backend\n");
        return false;
    }
    buffer.host = (uint8_t*)points.first + points.second;
    des->extra.offset = points.second;
    if (buffer.type.code == halide_type_handle) {
        // For handle we needn't recycle the buffer, use extra as hanleFreeFunction
        ::memset(buffer.host, 0, size);
        des->extra.handleFreeFunction = (decltype(des->extra.handleFreeFunction))free;
    }
    return true;
}

bool CPUBackend::onAcquireBuffer(const MNN::Tensor* nativeTensorConst, StorageType storageType) {
    if (nativeTensorConst == nullptr) {
        return false;
    }
    //FUNC_PRINT_ALL(nativeTensorConst, p);
    auto nativeTensor = (Tensor*)nativeTensorConst;
    auto size = nativeTensor->size();
    return allocBuffer(size, nativeTensor, storageType);
}

bool CPUBackend::onReleaseBuffer(const MNN::Tensor* nativeTensor, StorageType storageType) {
    if (DYNAMIC_SEPERATE == storageType) {
        return true;
    }
    if (nativeTensor == nullptr) {
        return false;
    }
    if (nullptr == nativeTensor->buffer().host) {
        return false;
    }
    auto des = TensorUtils::getDescribe(nativeTensor);
    std::pair<void*, int> pointer;
    pointer.second = des->extra.offset;
    pointer.first = (uint8_t*)nativeTensor->buffer().host - des->extra.offset;
    if (STATIC == storageType) {
        mStaticAllocator->free(pointer);
        return true;
    }
    mDynamicAllocator->free(pointer);
    return true;
}

std::pair<float, bool> CPUBackend::onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const MNN::Op* op) {
    auto map  = gCreator;
    auto iter = map->find(op->type());
    if (iter == map->end()) {
        MNN_PRINT("Don't support type %s, %s\n", MNN::EnumNameOpType(op->type()), op->name()->c_str());
        return std::make_pair(0.0f, false);
    }
    // FIXME: Compute in future
    return std::make_pair(0.0f, false);
}

halide_type_t CPUBackend::getRunType(const Op* op, halide_type_t qtype, halide_type_t rtype) {
    auto otype = op->type();
    switch (otype) {
        case OpType_Convolution:
        case OpType_ConvolutionDepthwise:
            if (op->main_as_Convolution2D() && op->main_as_Convolution2D()->weight() != nullptr) {
                return rtype;
            } else {
                return qtype;
            }
        case OpType_ConvInt8:
        case OpType_DepthwiseConvInt8:
        // case OpType_Eltwise:
        case OpType_Raster:
            return qtype;
        case OpType_ReLU:
            // now just relu without slope support quant
            if ((op->main_as_Relu() == nullptr) || op->main_as_Relu()->slope() == 0.f) {
                return qtype;
            } else {
                return rtype;
            }
        /*
        case OpType_Pooling:
            // now just maxpool support quant
            if (op->main_as_Pool() && op->main_as_Pool()->type() == PoolType_MAXPOOL) {
                return qtype;
            } else {
                return defaultType;
            }
        */
        default:
            return rtype;
    }
}

OpType CPUBackend::getRealOpType(OpType opType, halide_type_t dataType) {
    // now just support int8
    if (dataType != halide_type_of<int8_t>()) {
        return opType;
    }
    switch (opType) {
        case OpType_Convolution:
            return OpType_ConvInt8;
        case OpType_ConvolutionDepthwise:
            return OpType_DepthwiseConvInt8;
        /*
        case OpType_Pooling:
            return OpType_PoolInt8;
        */
        // case OpType_Eltwise:
        //     // TODO: just support EltwiseAdd
        //     return OpType_EltwiseInt8;
        default:
            return opType;
    }
}
int CPUBackend::getTensorSize(const Tensor* tensor) const {
    auto core = mCoreFunctions;
    int dataSize = 1;
    auto des = TensorUtils::getDescribe(tensor);
    for (int i = 0; i < tensor->dimensions(); i++) {
        int currentDimSize = tensor->length(i);
        if (des->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && 1 == i) {
            currentDimSize = UP_DIV(currentDimSize, core->pack) * core->pack;
        }
        dataSize *= currentDimSize;
    }
    MNN_ASSERT(dataSize > 0);
    return dataSize;
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
    // get QuantType and RunType, default is float
    halide_type_t quantType = halide_type_of<float>();
    auto isQuant = OpCommonUtils::getQuantInfo(inputs);
    if (isQuant.first) {
        // if output hasnt scale, using output type
        if (TensorUtils::getDescribe(outputs[0])->quantAttr == nullptr && !outputs.empty()) {
            quantType = outputs[0]->getType();
        } else {
            quantType = TensorUtils::DataTypeToHalideType(isQuant.second);
        }
    }
    auto originType = outputs.empty() ? halide_type_of<float>() : outputs[0]->getType();
    auto runType = getRunType(op, quantType, originType);
    // TODO: rm this convert when merge diff datatyoe of op
    auto opType = op->type();
    if (isQuant.first) {
        opType = getRealOpType(opType, runType);
    }
    auto map  = gCreator;
    auto iter = map->find(opType);
    if (iter == map->end()) {
        MNN_PRINT("Don't support type [%s], %s\n", MNN::EnumNameOpType(op->type()), op->name()->c_str());
        return nullptr;
    }
    Execution* exe = nullptr;
    if (isQuant.first) {
        bool needCast = false;
        // judge is it need CastWrap
        if (OpType_Raster == opType) {
            inputs[0]->setType(TensorUtils::HaildeTypeToDataType(runType));
            for (const auto& r : TensorUtils::getDescribe(inputs[0])->regions) {
                needCast |= (r.origin->getType() != runType);
            }
        } else {
            for (int i = 0; i < inputs.size(); i++) {
                if (OpCommonUtils::opNeedContent(opType, i) && inputs[i]->getType() != halide_type_of<int>()) {
                    needCast |= (inputs[i]->getType() != runType);
                }
            }
        }
        // set output Tensor Type
        auto outputType = TensorUtils::HaildeTypeToDataType(runType);
        for (auto output : outputs) {
            if (output->getType() != runType) {
                output->setType(outputType);
                needCast = true;
            }
        }
        if (needCast) {
            exe = new CastWrapExecution(iter->second, op, this, inputs, outputs, runType);
        }
    }
    if (exe == nullptr) {
        exe = iter->second->onCreate(inputs, outputs, op, this);
    }
    if (nullptr == exe) {
        return nullptr;
    }
    return makePostWrapExectuion(exe);
}
Execution* CPUBackend::makePostWrapExectuion(Execution* execution) const {
    if (!mCheckNAN) {
        return execution;
    }
    return new CheckNANExecution(execution);
}


bool CPUBackend::onClearBuffer() {
    mDynamicAllocator->release(true);
    mCachedCastTensor.clear();
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
    if (srcBuffer.type != dstBuffer.type) {
        ErrorCode code = NO_ERROR;
        if (TensorUtils::getDescribe(srcTensor)->dimensionFormat != TensorUtils::getDescribe(dstTensor)->dimensionFormat) {
            std::unique_ptr<Tensor> wrapTensor;
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
            wrapTensor.reset(Tensor::create(srcTensor->shape(), dstTensor->getType(), nullptr, dimType));
            code = CPUCastCreator::cast(srcTensor, wrapTensor.get(), this);
            CPUTensorConverter::convert(wrapTensor.get(), dstTensor);
        } else {
            code = CPUCastCreator::cast(srcTensor, dstTensor, this);
        }
        if (NO_ERROR != code) {
            MNN_ERROR("Error in CPUBackend::onCopyBuffer:cast\n");
        }
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
void registerCPURuntimeCreator() {
    CPUBackend::initCreatorMap();
    registerCPUOps();
#ifdef MNN_SUPPORT_BF16
    registerBF16Backend();
#endif
    // TODO: Merge _initCoreFunction MNNFunctionInit and cpuinfo_arm_init
    MNNCoreFunctionInit();
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_CPU, new CPURuntimeCreator);
};
} // namespace MNN
