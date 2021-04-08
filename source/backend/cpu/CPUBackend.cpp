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
#include "compute/Int8FunctionsOpt.h"
#include "CPUCast.hpp"
#include "core/OpCommonUtils.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP
#include "backend/cpu/CPURuntime.hpp"
#if defined(ENABLE_ARMV82) && (defined(__ANDROID__) || defined(__aarch64__))
#include "backend/arm82/Arm82Backend.hpp"
#endif
#define MAX_THREAD_NUMBER 32
#define LARGE_MEMORY 1024 * 1024 * 500
#ifdef MNN_SUPPORT_BF16
#include "bf16/BF16Backend.hpp"
#endif

#define MNN_CPU_CHECK_NAN 1
namespace MNN {
void registerCPUOps();
#if defined(ENABLE_ARMV82) && (defined(__ANDROID__) || defined(__aarch64__))
struct cpuinfo_arm_isa gCPUInfo;
#endif

CPURuntime::CPURuntime(const Backend::Info& info) {
    mStaticAllocator.reset(new BufferAllocator(BufferAllocator::Allocator::createDefault()));
    mThreadNumber = info.numThread;
    mThreadNumber = std::max(1, mThreadNumber);
    mThreadNumber = std::min(mThreadNumber, MAX_THREAD_NUMBER);
    mPower   = BackendConfig::Power_Normal;
    mMemory  = BackendConfig::Memory_Normal;
    mPrecision = BackendConfig::Precision_Normal;
    mFlags = 0;
    mFlops = MNNGetCPUFlops(mThreadNumber);
#if defined(ENABLE_ARMV82) && (defined(__ANDROID__) || defined(__aarch64__))
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
    auto staticMemoryInMB = mStaticAllocator->totalSize() / 1024.0f / 1024.0f;
    return staticMemoryInMB;
}
Backend* CPURuntime::onCreate(const BackendConfig* config) const {
    auto precision = mPrecision;
    if (nullptr != config) {
        precision = config->precision;
    }
#if defined(ENABLE_ARMV82) && (defined(__ANDROID__) || defined(__aarch64__))
    if (mIsSupportFp16arith && precision == BackendConfig::Precision_Low) {
        return new Arm82Backend(this);
    }
#endif
#ifdef MNN_SUPPORT_BF16
    if (precision == BackendConfig::Precision_Low) {
        return new BF16Backend(this);
    }
#endif
    return new CPUBackend(this, precision);
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

CPUBackend::CPUBackend(const CPURuntime* runtime, BackendConfig::PrecisionMode precision, MNNForwardType type) : Backend(type) {
    mRuntime = runtime;
    mCheckNAN = runtime->mFlags == MNN_CPU_CHECK_NAN;
    std::shared_ptr<BufferAllocator::Allocator> defaultAlloc(BufferAllocator::Allocator::createRecurse(runtime->mStaticAllocator.get()));
    mDynamicAllocator.reset(new BufferAllocator(defaultAlloc));
    mStaticAllocator = runtime->mStaticAllocator;
    mPrecisionMode = precision;
    mCoreFunctions = MNNGetCoreFunctions();
}
bool CPUBackend::supportDot() const {
    return mRuntime->mIsSupportDot;
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
        MNN_ASSERT(false);
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
        case OpType_Eltwise:
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
        case OpType_Eltwise:
            // TODO: just support EltwiseAdd
            return OpType_EltwiseInt8;
        default:
            return opType;
    }
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
            class CastWrapExecution : public Execution {
            public:
                CastWrapExecution(Backend* backend, halide_type_t runT, const Op* op, std::map<const Tensor*, const Tensor*>& cachedCastTensor, Execution* exe)
                    : Execution(backend), runType(runT), mOp(op), mCachedCastTensor(cachedCastTensor), mExecution(exe) {}
                CastWrapExecution(const CPUBackend::Creator* creator, const Op* op, Backend* backend,
                              const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                              halide_type_t runT, std::map<const Tensor*, const Tensor*>& cachedCastTensor)
                            : Execution(backend), runType(runT), mCreator(creator), mOp(op),
                              mCachedCastTensor(cachedCastTensor), mInputs(inputs) {
                    std::vector<int> types(inputs.size());
                    for (int i = 0; i < inputs.size(); i++) {
                        types[i] = TensorUtils::HaildeTypeToDataType(inputs[i]->getType());
                        inputs[i]->setType(TensorUtils::HaildeTypeToDataType(runType));
                    }
                    mExecution.reset(mCreator->onCreate(inputs, outputs, mOp, backend));
                    for (int i = 0; i < inputs.size(); i++) {
                        inputs[i]->setType(types[i]);
                    }
                }
                virtual ErrorCode onResize(const std::vector<Tensor*>& inputs,
                                           const std::vector<Tensor*>& outputs) override {
                    for (auto output : outputs) {
                        output->setType(TensorUtils::HaildeTypeToDataType(runType));
                    }
                    mWrapInputTensors.clear();
                    mWrapInputs.clear();
                    mCasts.clear();
                    mScales.clear();
                    std::vector<Tensor*> realInput;
                    if (mOp->type() == OpType_Raster) {
                        for (const auto& r : TensorUtils::getDescribe(inputs[0])->regions) {
                            realInput.push_back(r.origin);
                        }
                    } else {
                        realInput = inputs;
                    }
                    for (int i = 0; i < realInput.size(); i++) {
                        auto input = realInput[i];
                        if (input->getType() == runType || !OpCommonUtils::opNeedContent(mOp->type(), i) || input->getType() == halide_type_of<int>()) {
                            mWrapInputs.push_back(input);
                            continue;
                        }
                        if (mCachedCastTensor.find(input) != mCachedCastTensor.end()) {
                            mWrapInputs.push_back(const_cast<Tensor*>(mCachedCastTensor[input]));
                            continue;
                        }
                        std::unique_ptr<Tensor> wrapTensor(new Tensor);
                        TensorUtils::copyShape(input, wrapTensor.get(), true);
                        TensorUtils::getDescribe(wrapTensor.get())->quantAttr = TensorUtils::getDescribe(input)->quantAttr;
                        wrapTensor->buffer().type = runType;
                        bool memoryAllocSuccess = backend()->onAcquireBuffer(wrapTensor.get(), Backend::DYNAMIC);
                        if (!memoryAllocSuccess) {
                            return {};
                        }
                        mWrapInputs.push_back(wrapTensor.get());
                        auto wrapPointer = wrapTensor.get();
                        mCasts.insert(std::make_pair(input, wrapTensor.get()));
                        mCachedCastTensor.insert(std::make_pair(input, wrapTensor.get()));
                        mWrapInputTensors.emplace_back(std::move(wrapTensor));
                        mScales[input] = std::vector<float>(4);
                        auto& quantAttr = TensorUtils::getDescribe(input)->quantAttr;
                        float scale = runType == halide_type_of<float>() ? quantAttr->scale : 1/quantAttr->scale;
                        // set 4xscale for SSE compute
                        mScales[input][0] = scale;
                        mScales[input][1] = scale;
                        mScales[input][2] = scale;
                        mScales[input][3] = scale;
                    }
                    ErrorCode res = NO_ERROR;
                    if (mOp->type() == OpType_Raster) {
                        mRasterInput = inputs[0];
                        if (mCasts.size() > 0) {
                            mRasterInputTensor.reset(new Tensor(inputs[0], inputs[0]->getDimensionType(), false));
                            mRasterInput = mRasterInputTensor.get();
                            TensorUtils::getDescribe(mRasterInput)->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                            TensorUtils::getDescribe(mRasterInput)->regions.resize(realInput.size());
                            for (int i = 0; i < realInput.size(); i++) {
                                TensorUtils::getDescribe(mRasterInput)->regions[i] = TensorUtils::getDescribe(inputs[0])->regions[i];
                                TensorUtils::getDescribe(mRasterInput)->regions[i].origin = mWrapInputs[i];
                            }
                        }
                        res = mExecution->onResize({mRasterInput}, outputs);
                    } else {
                        res = mExecution->onResize(mWrapInputs, outputs);
                    }
                    for (auto& iter : mCasts) {
                        if (TensorUtils::getDescribe(iter.first)->useCount <= 1) {
                            backend()->onReleaseBuffer(iter.second, Backend::DYNAMIC);
                        }
                    }
                    return res;
                }

                virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs,
                                            const std::vector<Tensor*>& outputs) override {
                    for (const auto& iter : mCasts) {
                        auto input = iter.first;
                        auto output = iter.second;
                        auto& quantAttr = TensorUtils::getDescribe(input)->quantAttr;
                        MNN_ASSERT(quantAttr != nullptr);
                        auto numberThread = ((CPUBackend*)backend())->threadNumber();
                        if (numberThread == 1) {
                            CPUCastCreator::cast(input, output);
                            continue;
                        }
                        int size = input->elementSize();
                        int sizeQuad = size / 16;
                        int remain       = sizeQuad * 16;
                        int sizeDivide = sizeQuad / numberThread;
                        auto scale = mScales[input].data();
                        if (runType == halide_type_of<float>()) {
                            const auto inputDataPtr = input->host<int8_t>();
                            auto outputDataPtr      = output->host<float>();
                            if (sizeQuad > 0) {
                                MNN_CONCURRENCY_BEGIN(tId, numberThread) {
                                    int number = sizeDivide;
                                    if (tId == numberThread - 1) {
                                        number = sizeQuad - tId * sizeDivide;
                                    }
                                    const auto srcChannelPtr   = inputDataPtr + tId * sizeDivide * 16;
                                    auto dstChannlePtr         = outputDataPtr + tId * sizeDivide * 16;
                                    MNNInt8ScaleToFloat(dstChannlePtr, srcChannelPtr, scale, sizeDivide * 4, quantAttr->zero);
                                }
                                MNN_CONCURRENCY_END();
                            }
                            for (int i = remain; i < size; i++) {
                                outputDataPtr[i] = static_cast<int8_t>(std::min(std::max(inputDataPtr[i] * scale[0], quantAttr->min), quantAttr->max));
                            }
                        } else {
                            const auto inputDataPtr = input->host<float>();
                            auto outputDataPtr      = output->host<int8_t>();
                            if (sizeQuad > 0) {
                                MNN_CONCURRENCY_BEGIN(tId, numberThread) {
                                    int number = sizeDivide;
                                    if (tId == numberThread - 1) {
                                        number = sizeQuad - tId * sizeDivide;
                                    }
                                    const auto srcChannelPtr   = inputDataPtr + tId * sizeDivide * 16;
                                    auto dstChannlePtr         = outputDataPtr + tId * sizeDivide * 16;
                                    MNNFloat2Int8(srcChannelPtr, dstChannlePtr, sizeDivide * 4, scale, quantAttr->min, quantAttr->max, quantAttr->zero);
                                }
                                MNN_CONCURRENCY_END();
                            }
                            for (int i = remain; i < size; i++) {
                                outputDataPtr[i] = static_cast<float>(inputDataPtr[i]) * scale[0];
                            }
                        }
                    }
                    if (mOp->type() == OpType_Raster) {
                        return mExecution->onExecute({ mRasterInput }, outputs);
                    } else {
                        return mExecution->onExecute(mWrapInputs, outputs);
                    }
                }
                virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
                    if (dst == nullptr || bn == nullptr) {
                        return true;
                    }
                    Execution* exe;
                    mExecution->onClone(bn, op, &exe);
                    *dst = new CastWrapExecution(bn, runType, op, mCachedCastTensor, exe);
                    return true;
                };
            private:
                const Op* mOp;
                const CPUBackend::Creator* mCreator;
                halide_type_t runType;
                std::shared_ptr<Execution> mExecution;
                Tensor* mRasterInput;
                std::vector<Tensor*> mWrapInputs, mInputs;
                std::unique_ptr<Tensor> mRasterInputTensor;
                std::vector<std::unique_ptr<Tensor>> mWrapInputTensors;
                std::map<const Tensor*, const Tensor*> mCasts, &mCachedCastTensor;
                std::map<const Tensor*, std::vector<float>> mScales;
                bool firstResize = true;
            };
            exe = new CastWrapExecution(iter->second, op, this, inputs, outputs, runType, mCachedCastTensor);
        }
    }
    if (exe == nullptr) {
        exe = iter->second->onCreate(inputs, outputs, op, this);
    }
    if (nullptr == exe) {
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
                    if (TensorUtils::getDescribe(tensor)->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
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
        auto code = CPUCastCreator::cast(srcTensor, dstTensor);
        if (NO_ERROR != code) {
            MNN_ERROR("Error in CPUBackend::onCopyBuffer:cast\n");
            return;
        }
        srcTensor = dstTensor;
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
#if defined(ENABLE_ARMV82) && (defined(__ANDROID__) || defined(__aarch64__))
    cpuinfo_arm_init(&gCPUInfo);
#endif
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_CPU, new CPURuntimeCreator);
};
} // namespace MNN
