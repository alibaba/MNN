//
//  NeuronAdapterBackend.cpp
//  MNN
//
//  Created by MNN on 2021/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NeuronAdapterBackend.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"
#include <core/Macro.h>
#include <core/TensorUtils.hpp>
#include <stdlib.h>
#include <mutex>
#define MNN_OPEN_TIME_TRACE
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

#define NeuronAdapter_DEBUG_DEVICE
#define NeuronAdapter_DEBUG_OP
// #define NeuronAdapter_PROFILE
// #define USE_NCHW

#ifdef NeuronAdapter_DEBUG_OP
    #define NeuronAdapter_OP_LOG MNN_PRINT
#else
    #define NeuronAdapter_OP_LOG(...)
#endif
#ifdef NeuronAdapter_DEBUG_DEVICE
    #define NeuronAdapter_DEVICE_LOG MNN_PRINT
#else
    #define NeuronAdapter_DEVICE_LOG(...)
#endif
#define CHECK(func, ...)                                                \
  do {                                                                  \
    const auto _status = (func(__VA_ARGS__));                           \
    if (_status != NEURON_NO_ERROR) {                          \
      const auto ENUM_TO_STR = NeuronAdapterEnumToString(_status);              \
      MNN_ERROR("[NeuronAdapter] Error: %s when call " #func " at line %d.\n",  \
                                        ENUM_TO_STR.c_str(), __LINE__); \
      exit(0); \
    }                                                                   \
  } while (0)

namespace MNN {
    void registerNeuronAdapterOps();
    static inline std::map<OpType, NeuronAdapterBackend::Creator*>* getCreatorMap() {
        static std::once_flag of;
        static std::map<OpType, NeuronAdapterBackend::Creator*>* ret = nullptr;
        std::call_once(of, [&]() { ret = new std::map<OpType, NeuronAdapterBackend::Creator*>; });
        return ret;
    }

    std::string NeuronAdapterEnumToString(int code) {
        switch (code) {
#define ENUM_TO_STR(code) case NEURON_##code: return #code
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

    bool NeuronAdapterBackend::addCreator(OpType t, Creator* c) {
        auto map = getCreatorMap();
        if (map->find(t) != map->end()) {
            MNN_ERROR("Error: %d type has be added\n", t);
            return false;
        }
        map->insert(std::make_pair(t, c));
        return true;
    }

    NeuronAdapterBackend::NeuronAdapterBackend(const NeuronAdapterRuntime* runtime) : Backend(MNN_FORWARD_MTK_NEURON) {
        mNPURuntime = runtime;
        mPrecision  = mNPURuntime->mPrecision;
#ifdef USE_NCHW
        mNCHW = true;
#else
        mNCHW = false;
#endif
        NeuronAdapter_DEVICE_LOG("[NeuronAdapter] DimensionFormat is %s\n", mNCHW ? "NCHW" : "NHWC");
        if (mNeuronAdapterModel == nullptr) {
            CHECK(NeuronModel_create_27, &mNeuronAdapterModel);
        }
        if (mNeuronAdapterDevices.empty()) {
            uint32_t numDevices = 0;
            CHECK(Neuron_getDeviceCount_29, &numDevices);
            mNeuronAdapterDevices.resize(numDevices);
            NeuronAdapter_DEVICE_LOG("[NeuronAdapter] numDevices = %d\n", numDevices);
            for (int i = 0; i < numDevices; i++) {
                CHECK(Neuron_getDevice_29, i, &mNeuronAdapterDevices[i].device);
                CHECK(NeuronDevice_getName_29, mNeuronAdapterDevices[i].device, &mNeuronAdapterDevices[i].name);
                NeuronAdapter_DEVICE_LOG("[NeuronAdapter] device %d is : %s\n", i, mNeuronAdapterDevices[i].name);
            }
        }
    }

    NeuronAdapterBackend::~NeuronAdapterBackend() {
        NeuronCompilation_free_27(mNeuronAdapterCompilation);
        NeuronModel_free_27(mNeuronAdapterModel);
    }

    Execution* NeuronAdapterBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {
        auto map = getCreatorMap();
        auto iter = map->find(op->type());
        if (iter == map->end()) {
            MNN_ERROR("[NeuronAdapter] Don't support type %s.\n", MNN::EnumNameOpType(op->type()));
            return nullptr;
        }
        auto exe = iter->second->onCreate(inputs, outputs, op, this);
        if (nullptr == exe) {
            MNN_ERROR("[NeuronAdapter] The Creator Don't support type %s.\n", MNN::EnumNameOpType(op->type()));
            return nullptr;
        }
        NeuronAdapter_OP_LOG("[NeuronAdapter] Create op: %s, %s\n", MNN::EnumNameOpType(op->type()), op->name() ? op->name()->c_str() : "NoName");
        return exe;
    }

    void NeuronAdapterBackend::NeuronAdapterBackend::onExecuteBegin() const {
    }

    void NeuronAdapterBackend::onExecuteEnd() const {
        invokeModel();
    }

    Backend::MemObj* NeuronAdapterBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
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

    bool NeuronAdapterBackend::onClearBuffer() {
        mInputContentTensors.clear();
        mOutputContentTensors.clear();
        mOutputTensorIndexes.clear();
        mInputTensors.clear();
        mOutputTensors.clear();
        mInputIdxMap.clear();
        mOutputIdxMap.clear();
        return true;
    }

    void NeuronAdapterBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
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
                    MNN_ERROR("Error in NeuronAdapterBackend::onCopyBuffer:convert\n");
                }
                FLOAT_TO_HALF(tempTensor->host<float>(), realTensor->host<int16_t>(), realTensor->elementSize());
            } else {
                auto code = CPUTensorConverter::convert(srcTensor, realTensor);
                if (NO_ERROR != code) {
                    MNN_ERROR("Error in NeuronAdapterBackend::onCopyBuffer:convert\n");
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
                    MNN_ERROR("Error in NeuronAdapterBackend::onCopyBuffer:convert\n");
                }
            } else {
                auto code = CPUTensorConverter::convert(realTensor, dstTensor);
                if (NO_ERROR != code) {
                    MNN_ERROR("Error in NeuronAdapterBackend::onCopyBuffer:convert\n");
                }
            }
        }
    }

    void NeuronAdapterBackend::onResizeBegin() {
        mHalfBuffer.clear();
        mQuantCacheMap.clear();
    }

    ErrorCode NeuronAdapterBackend::onResizeEnd() {
        mOutputTensorIndexes.resize(mOutputContentTensors.size());
        for (int i=0; i<mOutputContentTensors.size(); ++i) {
            auto type = getDataType(mOutputTensors[i]);
            auto tensor = mOutputTensors[i];
            mOutputTensorIndexes[i] = getTensorIdx(tensor);
            if (DataType_DT_INT8 == type) {
                std::vector<uint32_t> udims;
                for (auto d : tensor->shape()) {
                    udims.push_back(d);
                }
                auto code = NEURON_TENSOR_FLOAT32;
                float scale = 0.0f;
                int zero = 0;
                dimsFormat<uint32_t>(udims, TensorUtils::getDescribe(tensor)->dimensionFormat);
                auto tmpIdx = buildOperand(nullptr, 0, code, udims, &scale, zero);
                mOutputTensorIndexes[i] = tmpIdx;
                buildOperation(NEURON_DEQUANTIZE, {getTensorIdx(tensor)}, {tmpIdx});
            }
        }
        buildModel();
        mHalfBuffer.clear();
        mQuantCacheMap.clear();
        return NO_ERROR;
    }
    uint32_t NeuronAdapterBackend::getTensorIdx(const Tensor* t, bool dequant) {
        //if (dequant) {
        //    const auto& qiter = mDequantIdxMap.find(t);
        //    if (qiter != mDequantIdxMap.end()) {
        //        return qiter->second;
        //    }
        //}
        const auto& iter = mTensorIdxMap.find(t);
        if (iter != mTensorIdxMap.end()) {
            return iter->second;
        }
        std::vector<uint32_t> udims;
        for (auto d : t->shape()) {
            udims.push_back(d);
        }
        dimsFormat<uint32_t>(udims, TensorUtils::getDescribe(t)->dimensionFormat);
        // scalar shape is {1} in NeuronAdapter
        if (udims.empty()) {
            udims.push_back(1);
        }
        float scale = 0.f;
        int zero = 0;
        auto dtype = t->getType();
        auto code = NEURON_TENSOR_FLOAT32;
        if (dtype == halide_type_of<int>()) {
            code = NEURON_TENSOR_INT32;
        } else if (dtype == halide_type_of<uint8_t>()) {
            code = NEURON_TENSOR_QUANT8_ASYMM;
            scale = 1.f;
        }
        if (TensorUtils::getDescribe(t)->quantAttr.get() != nullptr &&
        TensorUtils::getDescribe(t)->type == DataType_DT_INT8) {
            code = NEURON_TENSOR_QUANT8_ASYMM_SIGNED;
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
    ErrorCode NeuronAdapterBackend::replaceTensorWith(const Tensor* src, const Tensor* replace) {
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
        MNN_ERROR("[NeuronAdapter] The replace Tensor must register.");
        return INVALID_VALUE;
    }
    uint32_t NeuronAdapterBackend::buildDequantOperand(const Tensor* tensor) {
        const auto& iter = mDequantIdxMap.find(tensor);
        if (iter != mDequantIdxMap.end()) {
            return iter->second;
        }
        // 1. build tmp operand
        std::vector<uint32_t> udims;
        for (auto d : tensor->shape()) {
            udims.push_back(d);
        }
        auto code = NEURON_TENSOR_QUANT8_ASYMM_SIGNED;
        auto scale = TensorUtils::getDescribe(tensor)->quantAttr->scale;
        auto zero = TensorUtils::getDescribe(tensor)->quantAttr->zero;
        dimsFormat<uint32_t>(udims, TensorUtils::getDescribe(tensor)->dimensionFormat);
        auto tmpIdx = buildOperand(nullptr, 0, code, udims, &scale, zero);
        mDequantIdxMap.insert(std::make_pair(tensor, tmpIdx));
        mDequantMap.insert(std::make_pair(tmpIdx, tensor));
        return tmpIdx;
    }
    ErrorCode NeuronAdapterBackend::buildQuantOperation(const Tensor* src, const Tensor* dst) {
        auto srcIdx = getTensorIdx(src);
        const auto& iter = mQuantCacheMap.find(srcIdx);
        if (iter != mQuantCacheMap.end()) {
            // using cached quant output
            mTensorIdxMap.insert(std::make_pair(dst, iter->second));
            return NO_ERROR;
        }
        auto dstIdx = getTensorIdx(dst);
        mQuantCacheMap.insert(std::make_pair(srcIdx, dstIdx));
        return buildOperation(NEURON_QUANTIZE, {srcIdx}, {dstIdx});
    }
    uint32_t NeuronAdapterBackend::buildScalar(int scalar) {
        auto iter = mScalarIntMap.find(scalar);
        if (iter != mScalarIntMap.end()) {
            return iter->second;
        }
        auto scalarIdx = buildOperand(&scalar, 4, NEURON_INT32);
        mScalarIntMap.insert(std::make_pair(scalar, scalarIdx));
        return scalarIdx;
    }
    uint32_t NeuronAdapterBackend::buildScalar(bool scalar) {
        auto iter = mScalarBoolMap.find(scalar);
        if (iter != mScalarBoolMap.end()) {
            return iter->second;
        }
        uint8_t value = scalar;
        auto scalarIdx = buildOperand(&value, 1, NEURON_BOOL);
        mScalarBoolMap.insert(std::make_pair(scalar, scalarIdx));
        return scalarIdx;
    }
    uint32_t NeuronAdapterBackend::buildScalar(float scalar) {
        auto iter = mScalarFloatMap.find(scalar);
        if (iter != mScalarFloatMap.end()) {
            return iter->second;
        }
        uint32_t scalarIdx = -1;
        if (bytes() == 2) {
            uint16_t value = fp32to16(scalar);
            scalarIdx = buildOperand(&value, 2, NEURON_FLOAT16);
        } else {
            scalarIdx = buildOperand(&scalar, 4, NEURON_FLOAT32);
        }
        mScalarFloatMap.insert(std::make_pair(scalar, scalarIdx));
        return scalarIdx;
    }
    uint32_t NeuronAdapterBackend::buildOperand(const void* data, size_t size, int code, std::vector<uint32_t> dims, const float* scales, int zero) {
        bool useFP16 = (bytes() == 2 && code == NEURON_TENSOR_FLOAT32);
        if (useFP16) {
            code = NEURON_TENSOR_FLOAT16;
            size /= 2;
        }
        float scale = (scales && code != NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL) ? *scales : 0.f;
        NeuronOperandType operandType;
        operandType.type = code;
        operandType.dimensionCount = static_cast<uint32_t>(dims.size());
        operandType.dimensions = dims.empty() ? nullptr : dims.data();
        operandType.scale = scale;
        operandType.zeroPoint = zero;

        uint32_t operandIdx = mTensorIdx++;
        {
            NeuronAdapter_OP_LOG("build operand : {\n");
            NeuronAdapter_OP_LOG("\tidx : %d\n", operandIdx);
            NeuronAdapter_OP_LOG("\tdata : %p\n", data);
            NeuronAdapter_OP_LOG("\tsize : %d\n", size);
            NeuronAdapter_OP_LOG("\ttype : %d\n", operandType.type);
            NeuronAdapter_OP_LOG("\tscale : %f\n", scale);
            NeuronAdapter_OP_LOG("\tzero : %d\n", zero);
            NeuronAdapter_OP_LOG("\tdimensions : [ ");
            for (auto i : dims) NeuronAdapter_OP_LOG("%d, ", i);
            NeuronAdapter_OP_LOG("]\n}\n");
        }
        CHECK(NeuronModel_addOperand_27, mNeuronAdapterModel, &operandType);
        if (data && size) {
            if (useFP16) {
                mHalfBuffer.emplace_back(new int16_t[size/2]);
                FLOAT_TO_HALF(reinterpret_cast<const float*>(data), mHalfBuffer.back().get(), size/2);
                data = mHalfBuffer.back().get();
            }
            if (code == NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL) {
                MNN_ASSERT(scales != nullptr);
                NeuronSymmPerChannelQuantParams quantParam;
                quantParam.channelDim = 0;
                quantParam.scaleCount = dims[0];
                quantParam.scales = scales;
                NeuronModel_setOperandSymmPerChannelQuantParams_29(mNeuronAdapterModel, operandIdx, &quantParam);
            }
            CHECK(NeuronModel_setOperandValue_27, mNeuronAdapterModel, operandIdx, data, size);
        }
        return operandIdx;
    }
    ErrorCode NeuronAdapterBackend::buildOperation(NeuronOperationType op, const std::vector<uint32_t> &inputs, const std::vector<uint32_t> &outputs, const char* name) {
        {
            NeuronAdapter_OP_LOG("build operation : {\n");
            NeuronAdapter_OP_LOG("\tname : %s\n", name ? name : "none");
            NeuronAdapter_OP_LOG("\ttype : %d\n", op);
            NeuronAdapter_OP_LOG("\tinputs : [ ");
            for (auto i : inputs) NeuronAdapter_OP_LOG("%d, ", i);
            NeuronAdapter_OP_LOG("]\n\toutputs : [ ");
            for (auto i : outputs) NeuronAdapter_OP_LOG("%d, ", i);
            NeuronAdapter_OP_LOG("]\n}\n");
        }
        if (name) mOpNames.push_back(name);
        CHECK(NeuronModel_addOperation_27,
              mNeuronAdapterModel, op,
              inputs.size(), inputs.data(),
              outputs.size(), outputs.data());
//        for (auto output : outputs) {
//            const auto& iter = mDequantMap.find(output);
//            if (iter != mDequantMap.end()) {
//                // append dequant operation
//                buildOperation(NEURON_DEQUANTIZE, {output}, {getTensorIdx(iter->second)});
//            }
//        }
        return NO_ERROR;
    }

    ErrorCode NeuronAdapterBackend::buildModel() {
        // set input and output of model
        std::vector<uint32_t> inputOperands(mInputTensors.size()), outputOperands(mOutputTensors.size());
        for (int i = 0; i < mInputTensors.size(); i++) {
            inputOperands[i] = getTensorIdx(mInputTensors[i]);
        }

        {
            NeuronAdapter_OP_LOG("set model's inputs & outputs : {\n");
            NeuronAdapter_OP_LOG("\tinputs : [ ");
            for (auto i : inputOperands) NeuronAdapter_OP_LOG("%d, ", i);
            NeuronAdapter_OP_LOG("]\n\toutputs : [ ");
            for (auto i : mOutputTensorIndexes) NeuronAdapter_OP_LOG("%d, ", i);
            NeuronAdapter_OP_LOG("]\n}\n");
        }
        CHECK(NeuronModel_identifyInputsAndOutputs_27,
              mNeuronAdapterModel,
              inputOperands.size(),
              inputOperands.data(),
              mOutputTensorIndexes.size(),
              mOutputTensorIndexes.data());
        CHECK(NeuronModel_finish_27, mNeuronAdapterModel);
        std::unique_ptr<bool[]> supports(new bool[mOpNames.size()]);
        int selectDeviceIdx = -1;
        for (int i = 0; i < mNeuronAdapterDevices.size(); i++) {
            auto device = mNeuronAdapterDevices[i].device;
            auto name = mNeuronAdapterDevices[i].name;
            CHECK(NeuronModel_getSupportedOperationsForDevices_29, mNeuronAdapterModel, &device, 1, supports.get());
            NeuronAdapter_DEVICE_LOG("[NeuronAdapter] device [%d : %s] supportOps = {\n", i, name);
            bool allsupport = true;
            for (int i = 0; i < mOpNames.size(); i++) {
                allsupport &= supports[i];
                NeuronAdapter_DEVICE_LOG("\t%s : %d\n", mOpNames[i], supports[i]);
            }
            NeuronAdapter_DEVICE_LOG("}\n");
            if (allsupport) {
                selectDeviceIdx = i;
                break;
            }
        }
        MNN_PRINT("[NeuronAdapter] using device [%d : %s].\n", selectDeviceIdx, mNeuronAdapterDevices[selectDeviceIdx].name);
        CHECK(NeuronCompilation_createForDevices_29, mNeuronAdapterModel, &mNeuronAdapterDevices[selectDeviceIdx].device, 1, &mNeuronAdapterCompilation);
        CHECK(NeuronCompilation_setPreference_27, mNeuronAdapterCompilation, NEURON_PREFER_SUSTAINED_SPEED);
        MNN_PRINT("[NeuronAdapter] compilation start.\n");
        CHECK(NeuronCompilation_finish_27, mNeuronAdapterCompilation);
        MNN_PRINT("[NeuronAdapter] compilation end.\n");
        return NO_ERROR;
    }

    void NeuronAdapterBackend::invokeModel() const {
        NeuronExecution *execution;
        CHECK(NeuronExecution_create_27, mNeuronAdapterCompilation, &execution);
#ifdef NeuronAdapter_PROFILE
#endif
        for (int i = 0; i < mInputTensors.size(); i++) {
            const void* data = mInputContentTensors[i]->host<void>();
            size_t size = mInputContentTensors[i]->size();
            CHECK(NeuronExecution_setInput_27, execution, i, nullptr, data, size);
        }
        for (int i = 0; i < mOutputTensors.size(); i++) {
            void* data = mOutputContentTensors[i]->host<void>();
            size_t size = mOutputContentTensors[i]->size();
            CHECK(NeuronExecution_setOutput_27, execution, i, nullptr, data, size);
        }


        CHECK(NeuronExecution_compute_29, execution);
#ifdef NeuronAdapter_PROFILE
#endif
        NeuronExecution_free_27(execution);
    }

    NeuronAdapterRuntime::NeuronAdapterRuntime(const Backend::Info& info) {
        mInfo = info;
        BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
        BackendConfig::PowerMode power         = BackendConfig::Power_Normal;
        if (nullptr != mInfo.user) {
            precision = mInfo.user->precision;
            power     = mInfo.user->power;
        }

        mPrecision = precision;
    }

    NeuronAdapterRuntime::~NeuronAdapterRuntime() {}

    Backend* NeuronAdapterRuntime::onCreate(const BackendConfig* config, Backend* origin) const {
        return new NeuronAdapterBackend(this);
    }

    void NeuronAdapterRuntime::onGabageCollect(int level) {
        // nothing now
    }
    NeuronAdapterRuntime::CompilerType NeuronAdapterRuntime::onGetCompilerType() const {
        return Compiler_Geometry;
    }

    struct NeuronAdapterBackendCreator : RuntimeCreator {

        virtual Runtime* onCreate(const Backend::Info& info) const override {
            return new NeuronAdapterRuntime(info);
        }

        virtual bool onValid(Backend::Info& info) const override {
            return true;
        }
    };

    void registerNeuronAdapterRuntimeCreator() {
        if (!loadNeuronAdapterSymbol()) {
            return;
        }
        registerNeuronAdapterOps();
        MNNInsertExtraRuntimeCreator(MNN_FORWARD_MTK_NEURON, new NeuronAdapterBackendCreator, false);
    }
}
