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
// #define NNAPI_DEBUG
// #define USE_NCHW

#define CHECK(func, ...)                                                \
  do {                                                                  \
    const auto _status = (func(__VA_ARGS__));                           \
    if (_status != ANEURALNETWORKS_NO_ERROR) {                          \
      const auto ENUM_TO_STR = NNAPIEnumToString(_status);              \
      MNN_PRINT("[NNAPI] Error: %s when call " #func " at line %d.\n",  \
                                        ENUM_TO_STR.c_str(), __LINE__); \
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
    bool NNAPIBackend::addCreator(OpType t, Creator* c) {
        auto map = getCreatorMap();
        if (map->find(t) != map->end()) {
            MNN_PRINT("Error: %d type has be added\n", t);
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
        MNN_PRINT("[NNAPI] DimensionFormat is %s\n", mNCHW ? "NCHW" : "NHWC");
        if (mNNAPIModel == nullptr) {
            CHECK(ANeuralNetworksModel_create_27, &mNNAPIModel);
        }
        if (mNNAPIDevices.empty()) {
            uint32_t numDevices = 0;
            CHECK(ANeuralNetworks_getDeviceCount_29, &numDevices);
            mNNAPIDevices.resize(numDevices);
            MNN_PRINT("[NNAPI] numDevices = %d\n", numDevices);
            for (int i = 0; i < numDevices; i++) {
                CHECK(ANeuralNetworks_getDevice_29, i, &mNNAPIDevices[i].device);
                CHECK(ANeuralNetworksDevice_getName_29, mNNAPIDevices[i].device, &mNNAPIDevices[i].name);
                CHECK(ANeuralNetworksDevice_getType_29, mNNAPIDevices[i].device, &mNNAPIDevices[i].type);
                MNN_PRINT("[NNAPI] device %d is : %s, %d\n", i, mNNAPIDevices[i].name, mNNAPIDevices[i].type);
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
            MNN_PRINT("[NNAPI] Don't support type %s.\n", MNN::EnumNameOpType(op->type()));
            return nullptr;
        }
        auto exe = iter->second->onCreate(inputs, outputs, op, this);
        if (nullptr == exe) {
            MNN_PRINT("[NNAPI] The Creator Don't support type %s.\n", MNN::EnumNameOpType(op->type()));
            return nullptr;
        }
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
        std::unique_ptr<Tensor> tensor_(new Tensor(tensor, mNCHW ? Tensor::DimensionType::CAFFE : Tensor::DimensionType::TENSORFLOW, true));
        if(isInputCopy){
            mInputTensors.push_back(tensor);
            mInputContentTensors.push_back(std::move(tensor_));
            mInputIdxMap.insert(std::make_pair(tensor, mInputIdxMap.size()));
        }
        if(isOutputCopy){
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

        if(isConst){ return; }

        if (isInputCopy) {
            const auto iter = mInputIdxMap.find(dstTensor);
            MNN_ASSERT(iter != mInputIdxMap.end());
            // memcpy((void*)&mInputTensors[iter->second], &srcTensor, sizeof(void*));
            auto code = CPUTensorConverter::convert(srcTensor, mInputContentTensors[iter->second].get());
            if (NO_ERROR != code) {
                MNN_ERROR("Error in NNAPIBackend::onCopyBuffer:convert\n");
            }
        } else if (isOutputCopy) {
            const auto iter = mOutputIdxMap.find(srcTensor);
            MNN_ASSERT(iter != mOutputIdxMap.end());
            // memcpy(dstTensor->host<void>(), srcTensor->host<void>(), std::min(srcTensor->size(), dstTensor->size()));
            auto code = CPUTensorConverter::convert(mOutputContentTensors[iter->second].get(), dstTensor);
            if (NO_ERROR != code) {
                MNN_ERROR("Error in NNAPIBackend::onCopyBuffer:convert\n");
            }
        }
    }

    void NNAPIBackend::onResizeBegin() {
    }

    void NNAPIBackend::onResizeEnd() {
        buildModel();
    }

    uint32_t NNAPIBackend::getTensorIdx(const Tensor* t) {
        const auto& iter = mTensorIdxMap.find(t);
        if (iter != mTensorIdxMap.end()) {
            return iter->second;
        }
        std::vector<uint32_t> dims;
        for (auto d : t->shape()) {
            dims.push_back(d);
        }
        std::vector<uint32_t> udims(dims.begin(), dims.end());
        if (TensorUtils::getDescribe(t)->dimensionFormat != MNN_DATA_FORMAT_NHWC && !mNCHW) {
            // NCHW -> NHWC
            udims[0] = dims[0];
            udims[1] = dims[2];
            udims[2] = dims[3];
            udims[3] = dims[1];
        }
        uint32_t idx = buildOperand(nullptr, 0, ANEURALNETWORKS_TENSOR_FLOAT32, udims);
        mTensorIdxMap.insert(std::make_pair(t, idx));
        return idx;
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
        auto scalarIdx = buildOperand(&scalar, 4, ANEURALNETWORKS_FLOAT32);
        mScalarFloatMap.insert(std::make_pair(scalar, scalarIdx));
        return scalarIdx;
    }

    uint32_t NNAPIBackend::buildOperand(const void* data, size_t size, OperandCode code, std::vector<uint32_t> dims) {
        ANeuralNetworksOperandType operandType {
            .type = code,
            .dimensionCount = static_cast<uint32_t>(dims.size()),
            .dimensions = dims.empty() ? nullptr : dims.data(),
            .scale = 0.0f,
            .zeroPoint = 0,
        };
        CHECK(ANeuralNetworksModel_addOperand_27, mNNAPIModel, &operandType);
        uint32_t operandIdx = mTensorIdx++;
#ifdef NNAPI_DEBUG
        MNN_PRINT("build operand : {\n");
        MNN_PRINT("\tidx : %d\n", operandIdx);
        MNN_PRINT("\tdata : %p\n", data);
        MNN_PRINT("\tsize : %d\n", size);
        MNN_PRINT("\ttype : %d\n", operandType.type);
        MNN_PRINT("\tdimensions : [ ");
        for (auto i : dims) MNN_PRINT("%d, ", i);
        MNN_PRINT("]\n}\n");
#endif
        if (data && size) {
            CHECK(ANeuralNetworksModel_setOperandValue_27, mNNAPIModel, operandIdx, data, size);
        }
        return operandIdx;
    }

    ErrorCode NNAPIBackend::buildOperation(int op, const std::vector<uint32_t> &inputs, const std::vector<uint32_t> &outputs, const char* name) {
#ifdef NNAPI_DEBUG
        MNN_PRINT("build operation : {\n");
        MNN_PRINT("\ttype : %d\n", op);
        MNN_PRINT("\tinputs : [ ");
        for (auto i : inputs) MNN_PRINT("%d, ", i);
        MNN_PRINT("]\n\toutputs : [ ");
        for (auto i : outputs) MNN_PRINT("%d, ", i);
        MNN_PRINT("]\n}\n");
#endif
        if (name) mOpNames.push_back(name);
        CHECK(ANeuralNetworksModel_addOperation_27,
              mNNAPIModel, op,
              inputs.size(), inputs.data(),
              outputs.size(), outputs.data());
        return NO_ERROR;
    }

    void NNAPIBackend::buildModel() {
        // set input and output of model
        std::vector<uint32_t> inputOperands(mInputTensors.size()), outputOperands(mOutputTensors.size());
        for (int i = 0; i < mInputTensors.size(); i++) {
            inputOperands[i] = getTensorIdx(mInputTensors[i]);
        }
        for (int i = 0; i < mOutputTensors.size(); i++) {
            outputOperands[i] = getTensorIdx(mOutputTensors[i]);
        }
#ifdef NNAPI_DEBUG
        MNN_PRINT("set model's inputs & outputs : {\n");
        MNN_PRINT("\tinputs : [ ");
        for (auto i : inputOperands) MNN_PRINT("%d, ", i);
        MNN_PRINT("]\n\toutputs : [ ");
        for (auto i : outputOperands) MNN_PRINT("%d, ", i);
        MNN_PRINT("]\n}\n");
#endif
        CHECK(ANeuralNetworksModel_identifyInputsAndOutputs_27,
              mNNAPIModel,
              inputOperands.size(),
              inputOperands.data(),
              outputOperands.size(),
              outputOperands.data());
        // segment fault
        CHECK(ANeuralNetworksModel_finish_27, mNNAPIModel);
        std::unique_ptr<bool[]> supports(new bool[mOpNames.size()]);
        int selectDeviceIdx = -1;
        for (int i = 0; i < mNNAPIDevices.size(); i++) {
            auto device = mNNAPIDevices[i].device;
            auto name = mNNAPIDevices[i].name;
            auto type = mNNAPIDevices[i].type;
            CHECK(ANeuralNetworksModel_getSupportedOperationsForDevices_29, mNNAPIModel, &device, 1, supports.get());
            MNN_PRINT("[NNAPI] device [%d : %s] supportOps = {\n", i, name);
            bool allsupport = true;
            for (int i = 0; i < mOpNames.size(); i++) {
                allsupport &= supports[i];
                MNN_PRINT("\t%s : %d\n", mOpNames[i], supports[i]);
            }
            MNN_PRINT("}\n");
            if (allsupport) {
                selectDeviceIdx = i;
                MNN_PRINT("[NNAPI] using device [%d : %s : %d].\n", i, name, type);
                break;
            }
        }
        MNN_PRINT("[NNAPI] using device [%d : %s].\n", selectDeviceIdx, mNNAPIDevices[selectDeviceIdx].name);
        CHECK(ANeuralNetworksCompilation_createForDevices_29, mNNAPIModel, &mNNAPIDevices[selectDeviceIdx].device, 1, &mNNAPICompilation);
        CHECK(ANeuralNetworksCompilation_setPreference_27, mNNAPICompilation, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
        CHECK(ANeuralNetworksCompilation_finish_27, mNNAPICompilation);
        CHECK(ANeuralNetworksBurst_create_29, mNNAPICompilation, &mNNAPIBurst);
    }

    void NNAPIBackend::invokeModel() const {
// #define NNAPI_PROFILE
        ANeuralNetworksExecution *execution;
        CHECK(ANeuralNetworksExecution_create_27, mNNAPICompilation, &execution);
#ifdef NNAPI_PROFILE
        CHECK(ANeuralNetworksExecution_setMeasureTiming, execution, true);
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
#if 0
        ANeuralNetworksEvent *event = nullptr;
        CHECK(ANeuralNetworksExecution_startCompute, execution, &event);
        CHECK(ANeuralNetworksEvent_wait, event);
        ANeuralNetworksEvent_free(event);
#else
        CHECK(ANeuralNetworksExecution_compute_29, execution);
        // CHECK(ANeuralNetworksExecution_burstCompute_29, execution, mNNAPIBurst);
#endif
#ifdef NNAPI_PROFILE
        uint64_t duration;
        CHECK(ANeuralNetworksExecution_getDuration, execution, ANEURALNETWORKS_DURATION_IN_DRIVER, &duration);
        if (duration != UINT64_MAX) MNN_PRINT("[NNAPI] driver time : %f ms\n", duration / 1000000.0);
        CHECK(ANeuralNetworksExecution_getDuration, execution, ANEURALNETWORKS_DURATION_ON_HARDWARE, &duration);
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
        MNNInsertExtraRuntimeCreator(MNN_FORWARD_NN, new NNAPIBackendCreator, true);
    }
}
