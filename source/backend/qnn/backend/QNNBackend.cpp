//
//  QNNBackend.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNBackend.hpp"

namespace MNN {
namespace QNN {
// #define QNN_PROFILE_OP
// #define QNN_PROFILE_SUMMARIZE
// #define QNN_VORBOSE
QnnBackend::QnnBackend(const QnnRuntime* runtime) : Backend(MNNForwardType::MNN_FORWARD_NN), mPower(runtime->mPower) {
    mRuntime = runtime;
    mUseFP16 = (runtime->mPrecision != BackendConfig::Precision_High) ? true : false;
    mPerf = QNNPerf::create(&mRuntime->mQnnInterface);
    if (mPower == BackendConfig::Power_High) {
        mPerf->setPowerConfigBurst();
        mPerf->setRpcLatencyAndPolling();
    }

    // Set mQnnGraphConfig.
    mQnnHtpGraphCustomConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
    mQnnHtpGraphCustomConfig.precision = QNN_PRECISION_FLOAT16;
    mQnnGraphConfig.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    mQnnGraphConfig.customConfig = &mQnnHtpGraphCustomConfig;
}

QnnBackend::~QnnBackend() {
    clean();
    if (mPower == BackendConfig::Power_High) {
        mPerf->setPowerConfigBalanced();
    }
}

static inline std::map<OpType, QnnBackend::Creator*>* getCreatorMap() {
    static std::once_flag of;
    static std::map<OpType, QnnBackend::Creator*>* ret = nullptr;
    std::call_once(of, [&]() { ret = new std::map<OpType, QnnBackend::Creator*>; });
    return ret;
}

Execution* QnnBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {
    MNN_ASSERT(op != nullptr);
    auto map = getCreatorMap();
    auto iter = map->find(op->type());

    // MNN_PRINT("MNN_QNN::onCreate Type %d, Name %s.\n", op->type(), op->name()->c_str());

    if (iter == map->end()) {
        if(op->name() != nullptr){
            MNN_PRINT("MNN_QNN: Not registered type %d, %s.\n", op->type(), op->name()->c_str());
        } else {
            MNN_PRINT("MNN_QNN: Not registered type %d.\n", op->type());
        }
        return nullptr;
    }

    auto exe = iter->second->onCreate(inputs, outputs, op, this);

    if (nullptr == exe) {
        if(op->name() != nullptr){
            MNN_PRINT("MNN_QNN: Don't support type %d, %s.\n", op->type(), op->name()->c_str());
        } else {
            MNN_PRINT("MNN_QNN: Don't support type %d.\n", op->type());
        }
        return nullptr;
    }

    return exe;
}

bool QnnBackend::addCreator(OpType t, Creator* c) {
    auto map = getCreatorMap();
    if (map->find(t) != map->end()) {
        MNN_PRINT("MNN_QNN: %d type has be added.\n", t);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}


void QnnBackend::onExecuteBegin() const {
    if (mPower == BackendConfig::Power_Normal) {
        mPerf->setPowerConfigBurst();
        mPerf->setRpcLatencyAndPolling();
    }
    return;
}

#ifdef QNN_PROFILE_SUMMARIZE
static std::string getOpTypeFromName(const std::string& nodeName) {
    // The pattern is usually "OpType_..."
    size_t pos = nodeName.find('_');
    if (pos != std::string::npos) {
        return nodeName.substr(0, pos);
    }
    // Fallback for names without '_', like "Input OpId_2 (cycles)"
    pos = nodeName.find(' ');
    if (pos != std::string::npos) {
        return nodeName.substr(0, pos);
    }
    // If no delimiter is found, return the whole name as the type
    return nodeName;
}
#endif

void QnnBackend::startProfile() const{
#ifdef QNN_PROFILE_OP
    if (mQnnProfileHandle) {
        uint32_t numTopLevelEvents = 0;
        const QnnProfile_EventId_t* topLevelEvents = nullptr;

        auto get_err = mRuntime->mQnnInterface.profileGetEvents(mQnnProfileHandle, &topLevelEvents, &numTopLevelEvents);
        if (get_err != QNN_SUCCESS) {
            MNN_PRINT("[QNN Profile] Failed to get top-level events. Error: %d\n", (int)get_err);
            return;
        }

        MNN_PRINT("\n--- QNN Node-level Performance Report ---\n");
        bool foundNodeData = false;

        for (uint32_t i = 0; i < numTopLevelEvents; ++i) {
            QnnProfile_EventData_t eventData = QNN_PROFILE_EVENT_DATA_INIT;
            mRuntime->mQnnInterface.profileGetEventData(topLevelEvents[i], &eventData);

            if (eventData.type) {
                MNN_PRINT("Found EXECUTE event. Total time: %llu us. Querying sub-events...\n", (unsigned long long)eventData.value);
                
                uint32_t numSubEvents = 0;
                const QnnProfile_EventId_t* subEvents = nullptr;

                // 3. GetSubEvents
                auto get_sub_err = mRuntime->mQnnInterface.profileGetSubEvents(topLevelEvents[i], &subEvents, &numSubEvents);
                if (get_sub_err != QNN_SUCCESS) {
                    MNN_PRINT("[QNN Profile] Failed to get sub-events for EXECUTE event. Error: %d\n", (int)get_sub_err);
                    continue;
                }

                for (uint32_t j = 0; j < numSubEvents; ++j) {
                    QnnProfile_EventData_t subEventData = QNN_PROFILE_EVENT_DATA_INIT;
                    mRuntime->mQnnInterface.profileGetEventData(subEvents[j], &subEventData);

                    if (subEventData.type == QNN_PROFILE_EVENTTYPE_NODE) {
                        foundNodeData = true;
                        const char* nodeName = subEventData.identifier;
                        uint64_t value = subEventData.value;

                        switch (subEventData.unit) {
                            case QNN_PROFILE_EVENTUNIT_MICROSEC:
                                MNN_PRINT("Node: %-45s | Time: %10llu us (%.3f ms)\n", 
                                        nodeName, (unsigned long long)value, (double)value / 1000.0);
                                break;
                            case QNN_PROFILE_EVENTUNIT_CYCLES:
                                MNN_PRINT("Node: %-45s | Cycles: %.2f*10^6\n", nodeName, (double)value / 1000000.0);
                                break;
                            // ... other dealing ...
                            default:
                                MNN_PRINT("Node: %-45s | Value: %10llu (Unit: %u - Unknown)\n",
                                        nodeName, (unsigned long long)value, subEventData.unit);
                                break;
                        }
                    }
                }
            }
        }

        if (!foundNodeData) {
            MNN_PRINT("No node-specific performance data found. Please ensure you have set:\n");
            MNN_PRINT("1. Profile level to QNN_PROFILE_LEVEL_DETAILED.\n");
            MNN_PRINT("2. HTP graph config with QNN_HTP_GRAPH_CONFIG_OPTION_PERF_PROFILE (if available).\n");
        }
        MNN_PRINT("-----------------------------------------\n");
    }
#endif

#ifdef QNN_PROFILE_SUMMARIZE
    if (mQnnProfileHandle) {
        std::map<std::string, uint64_t> opCycleStats;
        uint64_t totalNodeCycles = 0;

        uint32_t numTopLevelEvents = 0;
        const QnnProfile_EventId_t* topLevelEvents = nullptr;

        auto get_err = mRuntime->mQnnInterface.profileGetEvents(mQnnProfileHandle, &topLevelEvents, &numTopLevelEvents);
        if (get_err != QNN_SUCCESS) {
            MNN_PRINT("[QNN Profile] Failed to get top-level events. Error: %d\n", (int)get_err);
            return;
        }

        for (uint32_t i = 0; i < numTopLevelEvents; ++i) {
            QnnProfile_EventData_t eventData = QNN_PROFILE_EVENT_DATA_INIT;
            mRuntime->mQnnInterface.profileGetEventData(topLevelEvents[i], &eventData);

            if (eventData.type) { // == QNN_PROFILE_EVENTTYPE_EXECUTE) {
                uint32_t numSubEvents = 0;
                const QnnProfile_EventId_t* subEvents = nullptr;
                auto get_sub_err = mRuntime->mQnnInterface.profileGetSubEvents(topLevelEvents[i], &subEvents, &numSubEvents);
                if (get_sub_err != QNN_SUCCESS) continue;

                for (uint32_t j = 0; j < numSubEvents; ++j) {
                    QnnProfile_EventData_t subEventData = QNN_PROFILE_EVENT_DATA_INIT;
                    mRuntime->mQnnInterface.profileGetEventData(subEvents[j], &subEventData);

                    if (subEventData.type == QNN_PROFILE_EVENTTYPE_NODE) {
                        if (subEventData.identifier) {
                            std::string opType = getOpTypeFromName(subEventData.identifier);
                            opCycleStats[opType] += subEventData.value;
                            totalNodeCycles += subEventData.value;
                        }
                    }
                }
            }
        }
        
        if (!opCycleStats.empty()) {
            MNN_PRINT("\n--- QNN Operator-wise Performance Summary ---\n");
            MNN_PRINT("%-20s | %15s | %s\n", "Operator Type", "Total Cycles", "Percentage");
            MNN_PRINT("--------------------------------------------------\n");

            std::vector<std::pair<std::string, uint64_t>> sortedStats(opCycleStats.begin(), opCycleStats.end());
            std::sort(sortedStats.begin(), sortedStats.end(), [](const std::pair<std::string, uint64_t>& a, const std::pair<std::string, uint64_t>& b) {
                return a.second > b.second; // sort by large -> small
            });

            for (const auto& pair : sortedStats) {
                double percentage = (totalNodeCycles > 0) ? ((double)pair.second * 100.0 / totalNodeCycles) : 0.0;
                MNN_PRINT("%-20s | %15llu | %.2f%%\n", pair.first.c_str(), pair.second, percentage);
            }
            MNN_PRINT("--------------------------------------------------\n");
            MNN_PRINT("%-20s | %15llu | 100.00%%\n", "Total", totalNodeCycles);
        }
    }
    // =========================================================
#endif
}

void QnnBackend::onExecuteEnd() const {
    executeGraph();
    if (mPower == BackendConfig::Power_Normal) {
        mPerf->setPowerConfigBalanced();
    }
    startProfile();
    return;
}

void QnnBackend::onResizeBegin() {
    clean();
    createContextAndGraph();
    return;
}

ErrorCode QnnBackend::onResizeEnd() {
    #ifdef QNN_VORBOSE
    MNN_PRINT("start finalize\n");
    #endif
    finalizeGraph();
    #ifdef QNN_VORBOSE
    MNN_PRINT("end finalize\n");
    #endif
    return NO_ERROR;
}


Backend::MemObj* QnnBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
    std::string tName = "QnnTensor_" + std::to_string(mTensorCounter);

    bool isInput = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
    bool isOutput = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
    bool isConst = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    MNN_ASSERT(!isConst);

    Qnn_TensorType_t tType = QNN_TENSOR_TYPE_NATIVE;
    if (isInput) {
        tType = QNN_TENSOR_TYPE_APP_WRITE;
    }
    if (isOutput) {
        tType = QNN_TENSOR_TYPE_APP_READ;
    }

    Qnn_DataType_t tDataType;
    MNN_ASSERT((tensor->getType().code == halide_type_float) || (tensor->getType().code == halide_type_int && tensor->getType().bits == 32));
    if (mUseFP16 && tensor->getType().code == halide_type_float) {
        tDataType = QNN_DATATYPE_FLOAT_16;
    } else if (tensor->getType().code == halide_type_float) {
        tDataType = QNN_DATATYPE_FLOAT_32;
    } else if (tensor->getType().code == halide_type_int && tensor->getType().bits == 32) {
        tDataType = QNN_DATATYPE_INT_32;
    } else {
        MNN_PRINT("MNN_QNN: Not supported data type in <QnnBackend::onAcquire>.\n");
        return nullptr;
    }

    Qnn_QuantizeParams_t tQuantizeParams{};
    tQuantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    tQuantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    Qnn_ScaleOffset_t tScaleOffsetEncoding;
    tScaleOffsetEncoding.scale = 0.0f;
    tScaleOffsetEncoding.offset = 0;
    tQuantizeParams.scaleOffsetEncoding = tScaleOffsetEncoding;

    std::unique_ptr<Tensor> tempTensor(new Tensor(tensor, gQnnTensorDimType, false));
    std::vector<int> tDims;
    if (!(tempTensor->shape().empty())) {
        tDims = tempTensor->shape();
    } else {
        tDims = {1};
    }

    std::shared_ptr<QNNTensorWrapper> qnnTensorWrapper = QNNTensorWrapper::create(tName, tType, tDataType, tDims, tQuantizeParams);

    Qnn_Tensor_t * qnnTensor = qnnTensorWrapper->getNativeTensor();
    CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnnTensor));

    if (isInput) {
        mInputTensorIndexes.push_back(mTensorCounter);
        qnnTensorWrapper->alloc();
    }
    if (isOutput) {
        mOutputTensorIndexes.push_back(mTensorCounter);
        qnnTensorWrapper->alloc();
    }
    mQNNTensorWrappers.push_back(qnnTensorWrapper);
    mTensorMap.insert({TensorUtils::getDescribe(tensor), mTensorCounter});

    mTensorCounter += 1;
    #ifdef QNN_VORBOSE
    MNN_PRINT("Total qnn tensor count:%d\n", mTensorCounter);
    #endif
    return new Backend::MemObj();
}


bool QnnBackend::onClearBuffer() {
    return true;
}


void QnnBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    bool isInput = TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
    bool isOutput = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
    bool isConst = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT || TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    MNN_ASSERT(!isConst);

    MNN_ASSERT(isInput || isOutput);

    if (isInput) {
        inputIO(srcTensor, dstTensor);
    } else if (isOutput) {
        outputIO(srcTensor, dstTensor);
    } else {
        // Not support.
    }
}

void QnnBackend::inputIO(const Tensor* srcTensor, const Tensor* dstTensor) const {
    int dstIndex = getTensorIdx(dstTensor);
    std::shared_ptr<QNNTensorWrapper> dstQnnTensorWrapper = mQNNTensorWrappers[dstIndex];
    std::shared_ptr<Tensor> dstDataContainer = dstQnnTensorWrapper->getDataContainer();

    bool valid0 = srcTensor->getType().code == halide_type_float;
    bool valid1 = srcTensor->getType().code == halide_type_int && srcTensor->getType().bits == 32;

    // Currently, support float and int input only.
    MNN_ASSERT(valid0 || valid1);

    if (valid1) {
        auto code = CPUTensorConverter::convert(srcTensor, dstDataContainer.get());
        if (NO_ERROR != code) {
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer.\n");
        }
        return;
    }

    if (mUseFP16) {
        std::unique_ptr<Tensor> stageTensor(Tensor::create<float>(dstDataContainer->shape(), nullptr, TensorUtils::getDimType(dstDataContainer.get())));
        auto code = CPUTensorConverter::convert(srcTensor, stageTensor.get());
        if (NO_ERROR != code) {
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer.\n");
        }
        FLOAT_TO_HALF(stageTensor->host<float>(), dstDataContainer->host<int16_t>(), dstDataContainer->elementSize());
    } else {
        auto code = CPUTensorConverter::convert(srcTensor, dstDataContainer.get());
        if (NO_ERROR != code) {
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer.\n");
        }
    }
}

void QnnBackend::outputIO(const Tensor* srcTensor, const Tensor* dstTensor) const {
    int srcIndex = getTensorIdx(srcTensor);
    std::shared_ptr<QNNTensorWrapper> srcQnnTensorWrapper = mQNNTensorWrappers[srcIndex];
    std::shared_ptr<Tensor> srcDataContainer = srcQnnTensorWrapper->getDataContainer();

    // Currently, support float output only.
    bool valid0 = dstTensor->getType().code == halide_type_float;
    bool valid1 = dstTensor->getType().code == halide_type_int && dstTensor->getType().bits == 32;

    // Currently, support float and int input only.
    MNN_ASSERT(valid0 || valid1);

    if (valid1) {
        auto code = CPUTensorConverter::convert(srcDataContainer.get(), dstTensor);
        if (NO_ERROR != code) {
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer.\n");
        }
        return;
    }

    if (mUseFP16) {
        std::unique_ptr<Tensor> stageTensor(Tensor::create<float>(srcDataContainer->shape(), nullptr, TensorUtils::getDimType(srcDataContainer.get())));
        HALF_TO_FLOAT(srcDataContainer->host<int16_t>(), stageTensor->host<float>(), srcDataContainer->elementSize());
        auto code = CPUTensorConverter::convert(stageTensor.get(), dstTensor);
        if (NO_ERROR != code) {
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer.\n");
        }
    } else {
        auto code = CPUTensorConverter::convert(srcDataContainer.get(), dstTensor);
        if (NO_ERROR != code) {
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer.\n");
        }
    }
}

void QnnBackend::createContextAndGraph() {
    CALL_QNN(mRuntime->mQnnInterface.contextCreate(mRuntime->mQnnBackendHandle, mRuntime->mQnnDeviceHandle, mQnnContextConfig, &mQnnContextHandle));
    MNN_ASSERT(mQnnContextHandle != nullptr);

    const QnnGraph_Config_t * pGraphConfig[] = {&mQnnGraphConfig, nullptr};
    CALL_QNN(mRuntime->mQnnInterface.graphCreate(mQnnContextHandle, mQnnGraphName.c_str(), pGraphConfig, &mQnnGraphHandle));
    MNN_ASSERT(mQnnGraphHandle != nullptr);
}

void QnnBackend::finalizeGraph() {
    // [TODO] Fix this. Add the following branch for empty resize.
    if (mTensorCounter == 0) {
        return;
    }
    #ifdef QNN_VORBOSE
    MNN_PRINT("Total qnn tensor count:%d\n", mTensorCounter);
    #endif

    #if defined(QNN_PROFILE_SUMMARIZE) || defined(QNN_PROFILE_OP)
    if (mQnnProfileHandle == nullptr) {
        // set QNN_PROFILE_LEVEL_DETAILED
        QnnProfile_Level_t profileLevel = QNN_PROFILE_LEVEL_DETAILED;
        MNN_PRINT("[QNN Profile] Creating QNN Profile Handle with DETAILED level.\n");
        auto profile_err = mRuntime->mQnnInterface.profileCreate(mQnnContextHandle, profileLevel, &mQnnProfileHandle);
        if (profile_err != QNN_SUCCESS || mQnnProfileHandle == nullptr) {
            MNN_ERROR("[QNN Profile] Failed to create QNN Profile Handle, error: %d\n", (int)profile_err);
            mQnnProfileHandle = nullptr;
        }
    }
    #endif
    CALL_QNN(mRuntime->mQnnInterface.graphFinalize(mQnnGraphHandle, mQnnProfileHandle, mQnnSignalHandle));
}

void QnnBackend::executeGraph() const {
    std::vector<Qnn_Tensor_t> inputs;
    std::vector<Qnn_Tensor_t> outputs;
    for (int i = 0; i <  mInputTensorIndexes.size(); i++) {
        inputs.push_back(*(mQNNTensorWrappers[mInputTensorIndexes[i]]->getNativeTensor()));
    }
    for (int j = 0 ; j < mOutputTensorIndexes.size(); j++) {
        outputs.push_back(*(mQNNTensorWrappers[mOutputTensorIndexes[j]]->getNativeTensor()));
    }

    CALL_QNN(mRuntime->mQnnInterface.graphExecute(mQnnGraphHandle, inputs.data(), mInputTensorIndexes.size(), outputs.data(), mOutputTensorIndexes.size(), mQnnProfileHandle, mQnnSignalHandle));
}

void QnnBackend::freeContextAndGraph() {
    if (mTensorCounter != 0) {
        CALL_QNN(mRuntime->mQnnInterface.contextFree(mQnnContextHandle, nullptr));
        mQnnContextHandle = nullptr;
        mQnnGraphHandle = nullptr;
    }
}

void QnnBackend::addNodeToGraph(Qnn_OpConfigVersion_t version, const char* nodeName, const char* packageName, const char* nodeType, std::vector<Qnn_Param_t> & params, std::vector<Qnn_Tensor_t> & inputs, std::vector<Qnn_Tensor_t> & outputs) {
    MNN_ASSERT(nodeName != nullptr && packageName != nullptr && nodeType != nullptr && !(inputs.empty()) && !(outputs.empty()));

    Qnn_OpConfig_t opConfig = QNN_OPCONFIG_INIT;
    opConfig.version = version;
    opConfig.v1.name = nodeName;
    opConfig.v1.packageName = packageName;
    opConfig.v1.typeName = nodeType;
    opConfig.v1.numOfParams = params.size();
    opConfig.v1.params = params.data();
    opConfig.v1.numOfInputs = inputs.size();
    opConfig.v1.inputTensors = inputs.data();
    opConfig.v1.numOfOutputs = outputs.size();
    opConfig.v1.outputTensors = outputs.data();

    CALL_QNN(mRuntime->mQnnInterface.backendValidateOpConfig(mRuntime->mQnnBackendHandle, opConfig));

    CALL_QNN(mRuntime->mQnnInterface.graphAddNode(mQnnGraphHandle, opConfig));
}

int QnnBackend::getTensorIdx(const Tensor * tensor) const {
    const Tensor::InsideDescribe::NativeInsideDescribe * tensorKey = TensorUtils::getDescribe(tensor);
    auto iter = mTensorMap.find(tensorKey);
    int idx = -1;
    if (iter == mTensorMap.end()) {
        std::string tName = "QnnTensor_" + std::to_string(mTensorCounter);;
        if (TensorUtils::getDescribe(tensor)->usage != Tensor::InsideDescribe::Usage::CONSTANT) {
            MNN_PRINT("Tensor usage is %d.\n", (int) TensorUtils::getDescribe(tensor)->usage);
        }
        #ifdef QNN_VORBOSE
        MNN_PRINT("qnn tenor usage:%d, dimension:%d\n", TensorUtils::getDescribe(tensor)->usage, tensor->dimensions());
        #endif
        MNN_ASSERT(TensorUtils::getDescribe(tensor)->usage == Tensor::InsideDescribe::Usage::CONSTANT);
        MNN_ASSERT(tensor->dimensions() <= 2);
        std::vector<uint32_t> tDims = getNHWCShape(tensor);
        Qnn_DataType_t tDataType;
        std::shared_ptr<QNNTensorWrapper> qnnTensorWrapper;
        if (tensor->getType().code == halide_type_int && tensor->getType().bits == 32) {
            tDataType = QNN_DATATYPE_INT_32;
            qnnTensorWrapper = QNNTensorWrapper::createStaticTensor(tName, tDataType, tDims, tensor->host<int>());
        } else if (tensor->getType().code == halide_type_float) {
            tDataType = mUseFP16 ? QNN_DATATYPE_FLOAT_16 : QNN_DATATYPE_FLOAT_32;
            qnnTensorWrapper = QNNTensorWrapper::createStaticFloatTensor(tName, tDataType, tDims, tensor->host<float>());
        } else {
            MNN_ASSERT(false);
        }
        Qnn_Tensor_t * qnnTensor = qnnTensorWrapper->getNativeTensor();
        CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnnTensor));
        mQNNTensorWrappers.push_back(qnnTensorWrapper);
        mTensorMap.insert({tensorKey, mTensorCounter});
        idx = mTensorCounter;
        mTensorCounter += 1;
    } else {
        idx = iter->second;
    }
    return idx;
}

void QnnBackend::addStaticTensorToGraph(Qnn_Tensor_t * staticTensor) {
    MNN_ASSERT(staticTensor->v1.type == QNN_TENSOR_TYPE_STATIC);
    CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, staticTensor));
}

void QnnBackend::addStageTensorToGraph(Qnn_Tensor_t * stageTensor) {
    MNN_ASSERT(stageTensor->v1.type == QNN_TENSOR_TYPE_NATIVE);
    CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, stageTensor));
}

Qnn_Tensor_t * QnnBackend::getNativeTensor(const Tensor * tensor) {
    int idx = getTensorIdx(tensor);
    return mQNNTensorWrappers[idx]->getNativeTensor();
}

std::shared_ptr<QNNTensorWrapper> QnnBackend::getTensorWrapper(const Tensor * tensor) {
    const Tensor::InsideDescribe::NativeInsideDescribe * tensorKey = TensorUtils::getDescribe(tensor);
    auto iter = mTensorMap.find(tensorKey);
    MNN_ASSERT(iter != mTensorMap.end());
    return mQNNTensorWrappers[iter->second];
}

bool QnnBackend::getUseFP16() const {
    return mUseFP16;
}

void QnnBackend::clean() {
    if (mQnnProfileHandle) {
        mRuntime->mQnnInterface.profileFree(mQnnProfileHandle);
        mQnnProfileHandle = nullptr;
    }
    freeContextAndGraph(); // This function must be called first.
    mTensorCounter = 0;
    mQNNTensorWrappers.clear();
    mTensorMap.clear();
    mInputTensorIndexes.clear();
    mOutputTensorIndexes.clear();
}

QnnRuntime::QnnRuntime(const Backend::Info& info, QNN_INTERFACE_VER_TYPE qnnInterface, Qnn_LogHandle_t qnnLogHandle, Qnn_BackendHandle_t qnnBackendHandle, Qnn_DeviceHandle_t qnnDeviceHandle) {
    // MNN_PRINT("QnnRuntime is constructing.\n");
    mInfo = info;
    // Default setting
    mPower = BackendConfig::Power_Normal;
    mMemory = BackendConfig::Memory_Normal;
    mPrecision = BackendConfig::Precision_Normal;
    // User setting
    if (info.user != nullptr) {
        mPrecision = info.user->precision;
        mPower = info.user->power;
        mMemory = info.user->memory;
    }
    mQnnInterface = qnnInterface;
    mQnnLogHandle = qnnLogHandle;
    mQnnBackendHandle = qnnBackendHandle;
    mQnnDeviceHandle = qnnDeviceHandle;
}

QnnRuntime::~QnnRuntime() {
    // Free Device
    CALL_QNN(mQnnInterface.deviceFree(mQnnDeviceHandle));

    // Free Backend
    CALL_QNN(mQnnInterface.backendFree(mQnnBackendHandle));

    // Free Log
    CALL_QNN(mQnnInterface.logFree(mQnnLogHandle));
}

Backend* QnnRuntime::onCreate(const BackendConfig* config, Backend* origin) const {
    return new QnnBackend(this);
}

QnnRuntime* QnnRuntime::create(const Backend::Info& info) {
    // Create Interface.
    QNN_INTERFACE_VER_TYPE qnnInterface{};
    {
#ifndef ENABLE_QNN_CONVERT_MODE
        QnnInterface_t** interfaceProviders = nullptr;
        uint32_t numProviders = 0;
        if (QnnInterface_getProviders((const QnnInterface_t***)&interfaceProviders, &numProviders) != QNN_SUCCESS) {
            MNN_PRINT("MNN_QNN: Failed to call 'QnnInterface_getProviders'.\n");
            return nullptr;
        }
        if (interfaceProviders == nullptr) {
            MNN_PRINT("MNN_QNN: Failed to get interface providers: null interface providers received.\n");
            return nullptr;
        }
        if (numProviders == 0) {
            MNN_PRINT("MNN_QNN: Failed to get interface providers: 0 interface providers.\n");
            return nullptr;
        }
        bool foundValidInterface = false;
        for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
            if (QNN_API_VERSION_MAJOR == interfaceProviders[pIdx]->apiVersion.coreApiVersion.major &&
                QNN_API_VERSION_MINOR <= interfaceProviders[pIdx]->apiVersion.coreApiVersion.minor) {
                foundValidInterface = true;
                qnnInterface = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
                break;
            }
        }
        if (!foundValidInterface) {
            MNN_PRINT("MNN_QNN: Failed to find a valid interface.\n");
            return nullptr;
        }
#else
        qnnInterface = gQnnConvertorInterface;
#endif
    }

    // Create Log.
    Qnn_LogHandle_t logHandle = nullptr;
    {
        QnnLog_Callback_t logCallback = nullptr;
        if ((QNN_GET_ERROR_CODE(qnnInterface.logCreate(logCallback, QNN_LOG_LEVEL_ERROR, &logHandle)) != QNN_SUCCESS) ||
            (logHandle == nullptr)) {
            MNN_PRINT("MNN_QNN: Failed to initialize logging in the backend.\n");
            return nullptr;
        }
    }

    // Create Backend.
    Qnn_BackendHandle_t backendHandle = nullptr;
    {
        const QnnBackend_Config_t** backendConfig = nullptr;
        if ((QNN_GET_ERROR_CODE(qnnInterface.backendCreate(logHandle, backendConfig, &backendHandle)) != QNN_SUCCESS) ||
            (backendHandle == nullptr)) {
            MNN_PRINT("MNN_QNN: Failed to create the backend.\n");
            return nullptr;
        }
    }

    // // Register custom OpPackages.
    // {
    //     std::string opPackagePathCPU = "/data/local/tmp/libQnnHtpOpPackageExample_cpu.so";
    //     std::string opPackagePathHTP = "/data/local/tmp/libQnnHtpOpPackageExample_htp.so";
    //     std::string opPackageInterfaceProvider = "exampleInterfaceProvider";
    //     std::string opPackageTargetCPU = "CPU";
    //     std::string opPackageTargetHTP = "HTP";
    //     if (!QnnRuntime::registerCustomOpPackage(qnnInterface, backendHandle, opPackagePathCPU.c_str(), opPackageInterfaceProvider.c_str(), opPackageTargetCPU.c_str())) {
    //         MNN_PRINT("MNN_QNN: Failed to register Op Package: %s.\n", opPackagePathCPU.c_str());
    //         return nullptr;
    //     }
    //     if (!QnnRuntime::registerCustomOpPackage(qnnInterface, backendHandle, opPackagePathHTP.c_str(), opPackageInterfaceProvider.c_str(), opPackageTargetHTP.c_str())) {
    //         MNN_PRINT("MNN_QNN: Failed to register Op Package: %s.\n", opPackageTargetHTP.c_str());
    //         return nullptr;
    //     }
    // }

    // Create Device.
    Qnn_DeviceHandle_t deviceHandle = nullptr;
    {
        // Check whether the device API is supported.
        bool supportDevice = checkCapability(qnnInterface, QNN_PROPERTY_GROUP_DEVICE);
        if (supportDevice) {
            const QnnDevice_Config_t ** deviceConfig = nullptr;
            auto qnnStatus = qnnInterface.deviceCreate(logHandle, deviceConfig, &deviceHandle);
            if(qnnStatus != QNN_SUCCESS || (deviceHandle == nullptr)) {
                MNN_PRINT("MNN_QNN: Failed to create the device, error:%lu\n", (unsigned long)qnnStatus);
                return nullptr;
            }
        } else {
            MNN_PRINT("MNN_QNN: Not supporting device API.\n");
            return nullptr;
        }
    }

    return new QnnRuntime(info, qnnInterface, logHandle, backendHandle, deviceHandle);
}

// Do nothing
void QnnRuntime::onGabageCollect(int level) {}

Runtime::CompilerType QnnRuntime::onGetCompilerType() const {
    return Compiler_Origin;
}

bool QnnRuntime::onSetCachePath(const char* path, int mode) {
#ifdef ENABLE_QNN_CONVERT_MODE
    MNN_ASSERT(path != nullptr);
    QNNConvertor::OutputDir = std::string(path);
#endif
    return true;
}

bool QnnRuntime::registerCustomOpPackage(QNN_INTERFACE_VER_TYPE qnnInterface, Qnn_BackendHandle_t backendHandle, const std::string & path, const std::string & interfaceProvider, const std::string & target) {
    if (QNN_GET_ERROR_CODE(qnnInterface.backendRegisterOpPackage(backendHandle, path.c_str(), interfaceProvider.c_str(), target.c_str())) != QNN_SUCCESS) {
        MNN_PRINT("MNN_QNN: Failed to register the Op Package: %s.\n", path.c_str());
        return false;
    }
    return true;
}

class QnnRuntimeCreator : public RuntimeCreator {
public:
    virtual Runtime* onCreate(const Backend::Info& info) const {
        return QnnRuntime::create(info);
    }
    virtual bool onValid(Backend::Info& info) const {
        return true;
    }
};

} // end namespace QNN

void registerQNNRuntimeCreator() {
#ifndef ENABLE_QNN_CONVERT_MODE
    // check whether the qnn lib is available
    if (!QNN::loadQNNSymbol()) {
        return;
    }
#endif
    QNN::registerQNNOps();
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_NN, new QNN::QnnRuntimeCreator, false);
}

} // end namespace MNN
