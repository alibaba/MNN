//
//  NNAPIBackend.cpp
//  MNN
//
//  Created by MNN on 2021/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NNAPIBackend.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"
#include <core/Macro.h>
#include <core/TensorUtils.hpp>
#include <stdlib.h>
#include <mutex>
#include <MNN/AutoTime.hpp>

#ifdef MNN_USE_ARMV82
// FP32 <--> FP16 Function
#include "backend/arm82/Arm82OptFunc.hpp"
#define FLOAT_TO_HALF MNNQuantizeFP16
#define HALF_TO_FLOAT MNNDequantizeFP16
#else
#define FLOAT_TO_HALF(...)
#define HALF_TO_FLOAT(...)
#endif // MNN_USE_ARMV82

// #define NNAPI_DEBUG_DEVICE
// #define NNAPI_DEBUG_OP
// #define NNAPI_PROFILE
// #define USE_NCHW

#ifdef NNAPI_DEBUG_OP
    #define NNAPI_OP_LOG MNN_PRINT
#else
    #define NNAPI_OP_LOG(...)
#endif
#ifdef NNAPI_DEBUG_DEVICE
    #define NNAPI_DEVICE_LOG MNN_PRINT
#else
    #define NNAPI_DEVICE_LOG(...)
#endif
#define CHECK(func, ...)                                                \
  do {                                                                  \
    const auto _status = (func(__VA_ARGS__));                           \
    if (_status != ANEURALNETWORKS_NO_ERROR) {                          \
      const auto ENUM_TO_STR = NNAPIEnumToString(_status);              \
      MNN_ERROR("[NNAPI] Error: %s when call " #func " at line %d.\n",  \
                                        ENUM_TO_STR.c_str(), __LINE__); \
      exit(0); \
    }                                                                   \
  } while (0)

namespace MNN {
    void registerNNAPIOps();
    static inline std::map<OpType, NNAPIBackend::Creator*>* getCreatorMap() {
        static std::once_flag of;
        static std::map<OpType, NNAPIBackend::Creator*>* ret = nullptr;
        std::call_once(of, [&]() { ret = new std::map<OpType, NNAPIBackend::Creator*>; });
        return ret;
    }

    std::string NNAPIEnumToString(int code) {
        switch (code) {
#define ENUM_TO_STR(code) case ANEURALNETWORKS_##code: return #code
            // ResultCode begin
            ENUM_TO_STR(NO_ERROR);
            ENUM_TO_STR(OUT_OF_MEMORY);
            ENUM_TO_STR(INCOMPLETE);
            ENUM_TO_STR(UNEXPECTED_NULL);
            ENUM_TO_STR(BAD_DATA);
            ENUM_TO_STR(OP_FAILED);
            ENUM_TO_STR(BAD_STATE);
            ENUM_TO_STR(UNMAPPABLE);
            ENUM_TO_STR(OUTPUT_INSUFFICIENT_SIZE);
            ENUM_TO_STR(UNAVAILABLE_DEVICE);
            // ResultCode end
            default:
                return "UNKNOWN_ENUM";
#undef ENUM_TO_STR
        }
    }

    static uint16_t fp32to16(float val) {
        uint32_t x = *((uint32_t*)&val);
        uint16_t h = ((x>>16)&0x8000)|((((x&0x7f800000)-0x38000000)>>13)&0x7c00)|((x>>13)&0x03ff);
        return h;
    }

    bool NNAPIBackend::addCreator(OpType t, Creator* c) {
        auto map = getCreatorMap();
        if (map->find(t) != map->end()) {
            MNN_ERROR("Error: %d type has be added\n", t);
            return false;
        }
        map->insert(std::make_pair(t, c));
        return true;
    }

    NNAPIBackend::NNAPIBackend(const NNAPIRuntime* runtime) : Backend(MNN_FORWARD_NN) {
        mNPURuntime = runtime;
        mPrecision  = mNPURuntime->mPrecision;
#ifdef USE_NCHW
        mNCHW = true;
#else
        mNCHW = false;
#endif
        NNAPI_DEVICE_LOG("[NNAPI] DimensionFormat is %s\n", mNCHW ? "NCHW" : "NHWC");
        if (mNNAPIModel == nullptr) {
            CHECK(ANeuralNetworksModel_create_27, &mNNAPIModel);
        }
        if (mNNAPIDevices.empty()) {
            uint32_t numDevices = 0;
            CHECK(ANeuralNetworks_getDeviceCount_29, &numDevices);
            mNNAPIDevices.resize(numDevices);
            NNAPI_DEVICE_LOG("[NNAPI] numDevices = %d\n", numDevices);
            for (int i = 0; i < numDevices; i++) {
                CHECK(ANeuralNetworks_getDevice_29, i, &mNNAPIDevices[i].device);
                CHECK(ANeuralNetworksDevice_getName_29, mNNAPIDevices[i].device, &mNNAPIDevices[i].name);
                CHECK(ANeuralNetworksDevice_getType_29, mNNAPIDevices[i].device, &mNNAPIDevices[i].type);
                NNAPI_DEVICE_LOG("[NNAPI] device %d is : %s, %d\n", i, mNNAPIDevices[i].name, mNNAPIDevices[i].type);
            }
        }
    }

    NNAPIBackend::~NNAPIBackend() {
        ANeuralNetworksCompilation_free_27(mNNAPICompilation);
        ANeuralNetworksModel_free_27(mNNAPIModel);
    }

    Execution* NNAPIBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {
        auto map = getCreatorMap();
        auto iter = map->find(op->type());
        if (iter == map->end()) {
            MNN_ERROR("[NNAPI] Don't support type %s.\n", MNN::EnumNameOpType(op->type()));
            return nullptr;
        }
        auto exe = iter->second->onCreate(inputs, outputs, op, this);
        if (nullptr == exe) {
            MNN_ERROR("[NNAPI] The Creator Don't support type %s.\n", MNN::EnumNameOpType(op->type()));
            return nullptr;
        }
        NNAPI_OP_LOG("[NNAPI] Create op: %s, %s\n", MNN::EnumNameOpType(op->type()), op->name() ? op->name()->c_str() : "NoName");
        return exe;
    }

    void NNAPIBackend::NNAPIBackend::onExecuteBegin() const {
    }
    
    void NNAPIBackend::onExecuteEnd() const {
        invokeModel();
    }

    Backend::MemObj* NNAPIBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
        bool isInputCopy = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
        bool isOutputCopy = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
        if (!isInputCopy && !isOutputCopy) {
            return new Backend::MemObj;
        }
        std::unique_ptr<Tensor> tensor_;
        auto format = mNCHW ? Tensor::DimensionType::CAFFE : Tensor::DimensionType::TENSORFLOW;
        if (bytes() == 2 && tensor->getType() == halide_type_of<float>()) {
            // fp16
            const_cast<Tensor*>(tensor)->buffer().type = halide_type_t(halide_type_float, 16);
            tensor_.reset(new Tensor(tensor, format, true));
            const_cast<Tensor*>(tensor)->buffer().type = halide_type_of<float>();
        } else {
            // fp32
            tensor_.reset(new Tensor(tensor, format, true));
        }
        if (TensorUtils::getDescribe(tensor)->quantAttr.get()) {
            // int8
            TensorUtils::getDescribe(tensor_.get())->quantAttr = TensorUtils::getDescribe(tensor)->quantAttr;
        }
        if (isInputCopy) {
            mInputTensors.push_back(tensor);
            mInputContentTensors.push_back(std::move(tensor_));
            mInputIdxMap.insert(std::make_pair(tensor, mInputIdxMap.size()));
        }
        if (isOutputCopy) {
            if (TensorUtils::getDescribe(tensor)->quantAttr.get()) {
                buildDequantOperand(tensor);
            }
            mOutputTensors.push_back(tensor);
            mOutputContentTensors.push_back(std::move(tensor_));
            mOutputIdxMap.insert(std::make_pair(tensor, mOutputIdxMap.size()));
            // TensorUtils::getDescribe(tensor)->memoryType = Tensor::InsideDescribe::MEMORY_HOST;
            // const_cast<halide_buffer_t&>(tensor->buffer()).host = (uint8_t*)MNNMemoryAllocAlign(tensor->size(), MNN_MEMORY_ALIGN_DEFAULT);
            // MNN_ASSERT(tensor->buffer().host != nullptr);
        }
        getTensorIdx(tensor);
        // Don't need release
        return new Backend::MemObj;
    }

    bool NNAPIBackend::onClearBuffer() {
        mInputContentTensors.clear();
        mOutputContentTensors.clear();
        mInputTensors.clear();
        mOutputTensors.clear();
        mInputIdxMap.clear();
        mOutputIdxMap.clear();
        return true;
    }

    void NNAPIBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
        bool isInputCopy = TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
        bool isOutputCopy = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
        bool isConst = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT || TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;

        if (isConst) { return; }

        std::unique_ptr<Tensor> tempTensor;
        if (isInputCopy) {
            const auto iter = mInputIdxMap.find(dstTensor);
            MNN_ASSERT(iter != mInputIdxMap.end());
            auto realTensor = mInputContentTensors[iter->second].get();
            if (bytes() == 2) {
                tempTensor.reset(Tensor::create<float>(realTensor->shape(), nullptr, TensorUtils::getDimType(realTensor)));
                auto code = CPUTensorConverter::convert(srcTensor, tempTensor.get());
                if (NO_ERROR != code) {
                    MNN_ERROR("Error in NNAPIBackend::onCopyBuffer:convert\n");
                }
                FLOAT_TO_HALF(tempTensor->host<float>(), realTensor->host<int16_t>(), realTensor->elementSize());
            } else {
                auto code = CPUTensorConverter::convert(srcTensor, realTensor);
                if (NO_ERROR != code) {
                    MNN_ERROR("Error in NNAPIBackend::onCopyBuffer:convert\n");
                }
            }
        } else if (isOutputCopy) {
            const auto iter = mOutputIdxMap.find(srcTensor);
            MNN_ASSERT(iter != mOutputIdxMap.end());
            auto realTensor = mOutputContentTensors[iter->second].get();
            if (bytes() == 2) {
                tempTensor.reset(Tensor::create<float>(realTensor->shape(), nullptr, TensorUtils::getDimType(realTensor)));
                HALF_TO_FLOAT(realTensor->host<int16_t>(), tempTensor->host<float>(), realTensor->elementSize());
                auto code = CPUTensorConverter::convert(tempTensor.get(), dstTensor);
                if (NO_ERROR != code) {
                    MNN_ERROR("Error in NNAPIBackend::onCopyBuffer:convert\n");
                }
            } else {
                auto code = CPUTensorConverter::convert(realTensor, dstTensor);
                if (NO_ERROR != code) {
                    MNN_ERROR("Error in NNAPIBackend::onCopyBuffer:convert\n");
                }
            }
        }
    }

    void NNAPIBackend::onResizeBegin() {
        mHalfBuffer.clear();
        mQuantCacheMap.clear();
    }

    ErrorCode NNAPIBackend::onResizeEnd() {
        buildModel();
        mHalfBuffer.clear();
        mQuantCacheMap.clear();
        return NO_ERROR;
    }
    uint32_t NNAPIBackend::getTensorIdx(const Tensor* t, bool dequant) {
        if (dequant) {
            const auto& qiter = mDequantIdxMap.find(t);
            if (qiter != mDequantIdxMap.end()) {
                return qiter->second;
            }
        }
        const auto& iter = mTensorIdxMap.find(t);
        if (iter != mTensorIdxMap.end()) {
            return iter->second;
        }
        std::vector<uint32_t> udims;
        for (auto d : t->shape()) {
            udims.push_back(d);
        }
        dimsFormat<uint32_t>(udims, TensorUtils::getDescribe(t)->dimensionFormat);
        // scalar shape is {1} in NNAPI
        if (udims.empty()) {
            udims.push_back(1);
        }
        float scale = 0.f;
        int zero = 0;
        auto dtype = t->getType();
        auto code = ANEURALNETWORKS_TENSOR_FLOAT32;
        if (dtype == halide_type_of<int>()) {
            code = ANEURALNETWORKS_TENSOR_INT32;
        } else if (dtype == halide_type_of<uint8_t>()) {
            code = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
            scale = 1.f;
        }
        if (TensorUtils::getDescribe(t)->quantAttr.get() != nullptr &&
            t->getType() == halide_type_of<int8_t>()) {
            code = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED;
            scale = TensorUtils::getDescribe(t)->quantAttr->scale;
            zero = TensorUtils::getDescribe(t)->quantAttr->zero;
        }
        uint32_t idx = -1;
        if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::Usage::CONSTANT) {
            idx = buildOperand(t->host<void>(), t->size(), code, udims);
        } else {
            idx = buildOperand(nullptr, 0, code, udims, &scale, zero);
        }
        mTensorIdxMap.insert(std::make_pair(t, idx));
        return idx;
    }
    ErrorCode NNAPIBackend::replaceTensorWith(const Tensor* src, const Tensor* replace) {
        const auto& qiter = mDequantIdxMap.find(replace);
        if (qiter != mDequantIdxMap.end()) {
            mDequantIdxMap.insert(std::make_pair(src, qiter->second));
            return NO_ERROR;
        }
        const auto& iter = mTensorIdxMap.find(replace);
        if (iter != mTensorIdxMap.end()) {
            mTensorIdxMap.insert(std::make_pair(src, iter->second));
            return NO_ERROR;
        }
        MNN_ERROR("[NNAPI] The replace Tensor must register.");
        return INVALID_VALUE;
    }
    uint32_t NNAPIBackend::buildDequantOperand(const Tensor* tensor) {
        const auto& iter = mDequantIdxMap.find(tensor);
        if (iter != mDequantIdxMap.end()) {
            return iter->second;
        }
        // 1. build tmp operand
        std::vector<uint32_t> udims;
        for (auto d : tensor->shape()) {
            udims.push_back(d);
        }
        auto code = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED;
        auto scale = TensorUtils::getDescribe(tensor)->quantAttr->scale;
        auto zero = TensorUtils::getDescribe(tensor)->quantAttr->zero;
        dimsFormat<uint32_t>(udims, TensorUtils::getDescribe(tensor)->dimensionFormat);
        auto tmpIdx = buildOperand(nullptr, 0, code, udims, &scale, zero);
        mDequantIdxMap.insert(std::make_pair(tensor, tmpIdx));
        mDequantMap.insert(std::make_pair(tmpIdx, tensor));
        return tmpIdx;
    }
    ErrorCode NNAPIBackend::buildQuantOperation(const Tensor* src, const Tensor* dst) {
        auto srcIdx = getTensorIdx(src);
        const auto& iter = mQuantCacheMap.find(srcIdx);
        if (iter != mQuantCacheMap.end()) {
            // using cached quant output
            mTensorIdxMap.insert(std::make_pair(dst, iter->second));
            return NO_ERROR;
        }
        auto dstIdx = getTensorIdx(dst);
        mQuantCacheMap.insert(std::make_pair(srcIdx, dstIdx));
        return buildOperation(ANEURALNETWORKS_QUANTIZE, {srcIdx}, {dstIdx});
    }
    uint32_t NNAPIBackend::buildScalar(int scalar) {
        auto iter = mScalarIntMap.find(scalar);
        if (iter != mScalarIntMap.end()) {
            return iter->second;
        }
        auto scalarIdx = buildOperand(&scalar, 4, ANEURALNETWORKS_INT32);
        mScalarIntMap.insert(std::make_pair(scalar, scalarIdx));
        return scalarIdx;
    }
    uint32_t NNAPIBackend::buildScalar(bool scalar) {
        auto iter = mScalarBoolMap.find(scalar);
        if (iter != mScalarBoolMap.end()) {
            return iter->second;
        }
        uint8_t value = scalar;
        auto scalarIdx = buildOperand(&value, 1, ANEURALNETWORKS_BOOL);
        mScalarBoolMap.insert(std::make_pair(scalar, scalarIdx));
        return scalarIdx;
    }
    uint32_t NNAPIBackend::buildScalar(float scalar) {
        auto iter = mScalarFloatMap.find(scalar);
        if (iter != mScalarFloatMap.end()) {
            return iter->second;
        }
        uint32_t scalarIdx = -1;
        if (bytes() == 2) {
            uint16_t value = fp32to16(scalar);
            scalarIdx = buildOperand(&value, 2, ANEURALNETWORKS_FLOAT16);
        } else {
            scalarIdx = buildOperand(&scalar, 4, ANEURALNETWORKS_FLOAT32);
        }
        mScalarFloatMap.insert(std::make_pair(scalar, scalarIdx));
        return scalarIdx;
    }
    uint32_t NNAPIBackend::buildOperand(const void* data, size_t size, OperandCode code, std::vector<uint32_t> dims, const float* scales, int zero) {
        bool useFP16 = (bytes() == 2 && code == ANEURALNETWORKS_TENSOR_FLOAT32);
        if (useFP16) {
            code = ANEURALNETWORKS_TENSOR_FLOAT16;
            size /= 2;
        }
        float scale = (scales && code != ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL) ? *scales : 0.f;
        ANeuralNetworksOperandType operandType;
        operandType.type = code;
        operandType.dimensionCount = static_cast<uint32_t>(dims.size());
        operandType.dimensions = dims.empty() ? nullptr : dims.data();
        operandType.scale = scale;
        operandType.zeroPoint = zero;

        uint32_t operandIdx = mTensorIdx++;
        {
            NNAPI_OP_LOG("build operand : {\n");
            NNAPI_OP_LOG("\tidx : %d\n", operandIdx);
            NNAPI_OP_LOG("\tdata : %p\n", data);
            NNAPI_OP_LOG("\tsize : %d\n", size);
            NNAPI_OP_LOG("\ttype : %d\n", operandType.type);
            NNAPI_OP_LOG("\tscale : %f\n", scale);
            NNAPI_OP_LOG("\tzero : %d\n", zero);
            NNAPI_OP_LOG("\tdimensions : [ ");
            for (auto i : dims) NNAPI_OP_LOG("%d, ", i);
            NNAPI_OP_LOG("]\n}\n");
        }
        CHECK(ANeuralNetworksModel_addOperand_27, mNNAPIModel, &operandType);
        if (data && size) {
            if (useFP16) {
                mHalfBuffer.emplace_back(new int16_t[size/2]);
                FLOAT_TO_HALF(reinterpret_cast<const float*>(data), mHalfBuffer.back().get(), size/2);
                data = mHalfBuffer.back().get();
            }
            if (code == ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL) {
                MNN_ASSERT(scales != nullptr);
                ANeuralNetworksSymmPerChannelQuantParams quantParam;
                quantParam.channelDim = 0;
                quantParam.scaleCount = dims[0];
                quantParam.scales = scales;
                ANeuralNetworksModel_setOperandSymmPerChannelQuantParams_29(mNNAPIModel, operandIdx, &quantParam);
            }
            CHECK(ANeuralNetworksModel_setOperandValue_27, mNNAPIModel, operandIdx, data, size);
        }
        return operandIdx;
    }
    ErrorCode NNAPIBackend::buildOperation(int op, const std::vector<uint32_t> &inputs, const std::vector<uint32_t> &outputs, const char* name) {
        {
            NNAPI_OP_LOG("build operation : {\n");
            NNAPI_OP_LOG("\tname : %s\n", name ? name : "none");
            NNAPI_OP_LOG("\ttype : %d\n", op);
            NNAPI_OP_LOG("\tinputs : [ ");
            for (auto i : inputs) NNAPI_OP_LOG("%d, ", i);
            NNAPI_OP_LOG("]\n\toutputs : [ ");
            for (auto i : outputs) NNAPI_OP_LOG("%d, ", i);
            NNAPI_OP_LOG("]\n}\n");
        }
        if (name) mOpNames.push_back(name);
        CHECK(ANeuralNetworksModel_addOperation_27,
              mNNAPIModel, op,
              inputs.size(), inputs.data(),
              outputs.size(), outputs.data());
        for (auto output : outputs) {
            const auto& iter = mDequantMap.find(output);
            if (iter != mDequantMap.end()) {
                // append dequant operation
                buildOperation(ANEURALNETWORKS_DEQUANTIZE, {output}, {getTensorIdx(iter->second)});
            }
        }
        return NO_ERROR;
    }

    ErrorCode NNAPIBackend::buildModel() {
        // set input and output of model
        std::vector<uint32_t> inputOperands(mInputTensors.size()), outputOperands(mOutputTensors.size());
        for (int i = 0; i < mInputTensors.size(); i++) {
            inputOperands[i] = getTensorIdx(mInputTensors[i]);
        }
        for (int i = 0; i < mOutputTensors.size(); i++) {
            auto output = mOutputTensors[i];
            outputOperands[i] = getTensorIdx(mOutputTensors[i]);
        }
        {
            NNAPI_OP_LOG("set model's inputs & outputs : {\n");
            NNAPI_OP_LOG("\tinputs : [ ");
            for (auto i : inputOperands) NNAPI_OP_LOG("%d, ", i);
            NNAPI_OP_LOG("]\n\toutputs : [ ");
            for (auto i : outputOperands) NNAPI_OP_LOG("%d, ", i);
            NNAPI_OP_LOG("]\n}\n");
        }
        CHECK(ANeuralNetworksModel_identifyInputsAndOutputs_27,
              mNNAPIModel,
              inputOperands.size(),
              inputOperands.data(),
              outputOperands.size(),
              outputOperands.data());
        CHECK(ANeuralNetworksModel_finish_27, mNNAPIModel);
        std::unique_ptr<bool[]> supports(new bool[mOpNames.size()]);
        int selectDeviceIdx = -1;
        for (int i = 0; i < mNNAPIDevices.size(); i++) {
            auto device = mNNAPIDevices[i].device;
            auto name = mNNAPIDevices[i].name;
            auto type = mNNAPIDevices[i].type;
            CHECK(ANeuralNetworksModel_getSupportedOperationsForDevices_29, mNNAPIModel, &device, 1, supports.get());
            NNAPI_DEVICE_LOG("[NNAPI] device [%d : %s] supportOps = {\n", i, name);
            bool allsupport = true;
            for (int i = 0; i < mOpNames.size(); i++) {
                allsupport &= supports[i];
                NNAPI_DEVICE_LOG("\t%s : %d\n", mOpNames[i], supports[i]);
            }
            NNAPI_DEVICE_LOG("}\n");
            if (allsupport) {
                selectDeviceIdx = i;
                NNAPI_DEVICE_LOG("[NNAPI] using device [%d : %s : %d].\n", i, name, type);
                break;
            }
        }
        MNN_PRINT("[NNAPI] using device [%d : %s].\n", selectDeviceIdx, mNNAPIDevices[selectDeviceIdx].name);
        CHECK(ANeuralNetworksCompilation_createForDevices_29, mNNAPIModel, &mNNAPIDevices[selectDeviceIdx].device, 1, &mNNAPICompilation);
        CHECK(ANeuralNetworksCompilation_setPreference_27, mNNAPICompilation, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
        CHECK(ANeuralNetworksCompilation_finish_27, mNNAPICompilation);
        CHECK(ANeuralNetworksBurst_create_29, mNNAPICompilation, &mNNAPIBurst);
        return NO_ERROR;
    }

    void NNAPIBackend::invokeModel() const {
        ANeuralNetworksExecution *execution;
        CHECK(ANeuralNetworksExecution_create_27, mNNAPICompilation, &execution);
#ifdef NNAPI_PROFILE
        ANeuralNetworksExecution_setMeasureTiming_29(execution, true);
#endif
        for (int i = 0; i < mInputTensors.size(); i++) {
            const void* data = mInputContentTensors[i]->host<void>();
            size_t size = mInputContentTensors[i]->size();
            CHECK(ANeuralNetworksExecution_setInput_27, execution, i, nullptr, data, size);
        }
        for (int i = 0; i < mOutputTensors.size(); i++) {
            void* data = mOutputContentTensors[i]->host<void>();
            size_t size = mOutputContentTensors[i]->size();
            CHECK(ANeuralNetworksExecution_setOutput_27, execution, i, nullptr, data, size);
        }

        CHECK(ANeuralNetworksExecution_compute_29, execution);
#ifdef NNAPI_PROFILE
        uint64_t duration;
        ANeuralNetworksExecution_getDuration_29(execution, ANEURALNETWORKS_DURATION_IN_DRIVER, &duration);
        if (duration != UINT64_MAX) MNN_PRINT("[NNAPI] driver time : %f ms\n", duration / 1000000.0);
        ANeuralNetworksExecution_getDuration_29(execution, ANEURALNETWORKS_DURATION_ON_HARDWARE, &duration);
        if (duration != UINT64_MAX) MNN_PRINT("[NNAPI] hardware time : %f ms\n", duration / 1000000.0);
#endif
        ANeuralNetworksExecution_free_27(execution);
    }

    NNAPIRuntime::NNAPIRuntime(const Backend::Info& info) {
        mInfo = info;
        BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
        BackendConfig::PowerMode power         = BackendConfig::Power_Normal;
        if (nullptr != mInfo.user) {
            precision = mInfo.user->precision;
            power     = mInfo.user->power;
        }

        mPrecision = precision;
    }

    NNAPIRuntime::~NNAPIRuntime() {}

    Backend* NNAPIRuntime::onCreate(const BackendConfig* config) const {
        return new NNAPIBackend(this);
    }

    void NNAPIRuntime::onGabageCollect(int level) {
        // nothing now
    }
    NNAPIRuntime::CompilerType NNAPIRuntime::onGetCompilerType() const {
        return Compiler_Geometry;
    }

    struct NNAPIBackendCreator : RuntimeCreator {

        virtual Runtime* onCreate(const Backend::Info& info) const override {
            return new NNAPIRuntime(info);
        }

        virtual bool onValid(Backend::Info& info) const override {
            return true;
        }
    };

    void registerNNAPIRuntimeCreator() {
        if (!loadNNAPISymbol()) {
            return;
        }
        registerNNAPIOps();
        MNNInsertExtraRuntimeCreator(MNN_FORWARD_NN, new NNAPIBackendCreator, false);
    }
}
