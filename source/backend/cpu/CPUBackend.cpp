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




Backend* CPURuntime::onCreate(const BackendConfig* config) const {
    auto precision = mPrecision;
    size_t flags = mFlags;
    if (nullptr != config) {
        precision = config->precision;
        flags = config->flags;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("cpu backend was created by runtime:%p\n", this);
#endif

#ifdef MNN_USE_ARMV82
    auto core = MNNGetCoreFunctions();
    if (core->supportFp16arith && precision == BackendConfig::Precision_Low) {
        return new Arm82Backend(this);
    }
#endif
#ifdef MNN_SUPPORT_BF16
    if (precision == BackendConfig::Precision_Low_BF16 && BF16Functions::get()) {
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
    if (mThreadNumber > 1 && mPower != BackendConfig::Power_High) {
        mTaskIndex = ThreadPool::acquireWorkIndex();
        if (mTaskIndex >= 0 ) {
            ThreadPool::active();
        }
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
        ThreadPool::releaseWorkIndex(mTaskIndex);
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

CPUBackend::CPUBackend(const CPURuntime* runtime, BackendConfig::PrecisionMode precision, MNNForwardType type, size_t flags) : Backend(type) {
#ifdef LOG_VERBOSE
    MNN_PRINT("cpu backend create\n");
#endif
    mRuntime = const_cast<CPURuntime*>(runtime);
    std::shared_ptr<BufferAllocator::Allocator> defaultAlloc(BufferAllocator::Allocator::createRecurse(runtime->mStaticAllocator.get()));
    mDynamicAllocator.reset(new BufferAllocator(defaultAlloc));
    mStaticAllocator = runtime->mStaticAllocator;
    mPrecisionMode = precision;
    mCoreFunctions = MNNGetCoreFunctions();
    mInt8CoreFunctions = MNNGetInt8CoreFunctions();
    mCache = new CPUResizeCache;
}

CPUBackend::~CPUBackend() {
    delete mCache;
}

void CPUBackend::onExecuteBegin() const {
    mRuntime->onConcurrencyBegin();
}

void CPUBackend::onExecuteEnd() const {
    mRuntime->onConcurrencyEnd();
}

class CPUMemObj : public Backend::MemObj {
public:
    CPUMemObj(BufferAllocator* allocator, std::pair<void*, int> points, int size) {
        mPoint = std::move(points);
        mAllocator = allocator;
        mSize = size;
    }
    virtual ~ CPUMemObj() {
        mAllocator->free(mPoint);
    }
    inline int getSize() const {
        return mSize;
    }
private:
    BufferAllocator* mAllocator;
    std::pair<void*, int> mPoint;
    int mSize;
};

Backend::MemObj* CPUBackend::allocBuffer(int size, Tensor* dest, StorageType storageType) {
    auto originMem = TensorUtils::getDescribe(dest)->mem.get();
    if (nullptr != originMem) {
        if (static_cast<CPUMemObj*>(originMem)->getSize() >= size) {
            return originMem;
        } else {
            TensorUtils::getDescribe(dest)->mem.reset(nullptr);
        }
    }
    // MNN_PRINT("Acquire size = %d\n", size);
    if (size <= 0) {
        MNN_PRINT("Acquire buffer size = %d\n", size);
       // MNN_ASSERT(false);
        return nullptr;
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
        return nullptr;
    }
    Backend::MemObj* res = nullptr;
    if (storageType == STATIC) {
        res = new CPUMemObj(mStaticAllocator.get(), points, size);
    } else {
        res = new CPUMemObj(mDynamicAllocator.get(), points, size);
    }
    buffer.host = (uint8_t*)points.first + points.second;
    des->extra.offset = points.second;
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

static bool _supportQuant(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto otype = op->type();
    switch (otype) {
        case OpType_Convolution:
        case OpType_ConvolutionDepthwise:
            if (op->main_as_Convolution2D() && op->main_as_Convolution2D()->weight() != nullptr) {
                return false;
            } else {
                return true;
            }
        case OpType_ConvInt8:
        case OpType_DepthwiseConvInt8:
            return true;
        // case OpType_Eltwise:
        case OpType_Raster:
        {
            for (auto& r : TensorUtils::getDescribe(inputs[0])->regions) {
                if (TensorUtils::getDescribe(r.origin)->quantAttr.get() != TensorUtils::getDescribe(outputs[0])->quantAttr.get()) {
                    return false;
                }
            }
            return true;
        }
        case OpType_ReLU:
            if (TensorUtils::getDescribe(inputs[0])->quantAttr.get() != TensorUtils::getDescribe(outputs[0])->quantAttr.get()) {
                return false;
            }
            // now just relu without slope support quant
            if ((op->main_as_Relu() == nullptr) || op->main_as_Relu()->slope() == 0.f) {
                return true;
            } else {
                return false;
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
            return false;
    }
    return false;
}

static OpType _getRealOpType(OpType opType) {
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
int CPUBackend::getTensorSize(const Tensor* tensor, bool multiBytes) const {
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
    if (multiBytes) {
        int bytes = tensor->getType().bytes();
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
    // Check if need use quant op
    DataType runType = DataType_DT_FLOAT;
    bool useQuant = false;
    if (outputs.size() == 1) {
        // Quant: output and all input has quantAttr and op support
        if (TensorUtils::getDescribe(outputs[0])->quantAttr != nullptr) {
            useQuant = _supportQuant(op, inputs, outputs);
        }
        if (useQuant) {
            if (op->type() == OpType_Raster) {
                for (auto& t : TensorUtils::getDescribe(inputs[0])->regions) {
                    if (TensorUtils::getDescribe(t.origin)->quantAttr == nullptr || TensorUtils::getDescribe(t.origin)->type == DataType_DT_FLOAT) {
                        useQuant = false;
                        break;
                    }
                }
            } else {
                for (auto t : inputs) {
                    if (TensorUtils::getDescribe(t)->quantAttr == nullptr) {
                        useQuant = false;
                        break;
                    }
                }
            }
        }
    }
    auto opType = op->type();
    if (useQuant) {
        opType = _getRealOpType(opType);
        runType = DataType_DT_INT8;
        TensorUtils::getDescribe(outputs[0])->type = DataType_DT_INT8;
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
    // judge is it need CastWrap
    if (OpType_Raster == opType) {
        TensorUtils::getDescribe(inputs[0])->quantAttr = TensorUtils::getDescribe(outputs[0])->quantAttr;
        for (const auto& r : TensorUtils::getDescribe(inputs[0])->regions) {
            needCast |= getDataType(r.origin) != runType;
        }
    } else {
        for (int i = 0; i < inputs.size(); i++) {
            if (OpCommonUtils::opNeedContent(opType, i) && inputs[i]->getType() != halide_type_of<int>()) {
                needCast |= getDataType(inputs[i]) != runType;
            }
        }
    }
    if (needCast) {
        exe = new CastWrapExecution(iter->second, op, this, inputs, outputs, runType);
    }
    if (exe == nullptr) {
        exe = iter->second->onCreate(inputs, outputs, op, this);
    }
    for (auto o : outputs) {
        auto quan = TensorUtils::getDescribe(o)->quantAttr;
        if (nullptr != quan) {
            TensorUtils::getDescribe(o)->type = runType;
        }
    }
    if (nullptr == exe) {
        return nullptr;
    }
    return exe;
}
const Runtime* CPUBackend::getRuntime() {
    return mRuntime;
}

bool CPUBackend::onClearBuffer() {
    mCache->reset();
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
    std::unique_ptr<Tensor> wrapTensor;
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
    CPUBackend::initCreatorMap();
    registerCPUOps();
#ifdef MNN_SUPPORT_BF16
    registerBF16Backend();
#endif
#ifdef MNN_USE_ARMV82
    registerArm82RuntimeCreator();
#endif
    // TODO: Merge _initCoreFunction MNNFunctionInit and cpuinfo_arm_init
    MNNCoreFunctionInit();
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_CPU, new CPURuntimeCreator);
};
} // namespace MNN
