#include "QNNConvertorInterface.hpp"
#include "QNNConvertor.hpp"

#define NOTNULL ((void *) 0x1)

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

extern QnnHtpDevice_PerfInfrastructure_t gQnnConvertorPerfInfra;
extern QnnHtpDevice_Infrastructure_t gQnnConvertorDeviceInfra;

Qnn_ErrorHandle_t QnnConvertorProperty_hasCapability(QnnProperty_Key_t key) {
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorLog_Create(QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel, Qnn_LogHandle_t* logger) {
    *logger = NOTNULL;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorLog_Free(Qnn_LogHandle_t logger) {
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorBackend_Create(Qnn_LogHandle_t logger, const QnnBackend_Config_t** config, Qnn_BackendHandle_t* backend) {
    *backend = NOTNULL;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorBackend_RegisterOpPackage(Qnn_BackendHandle_t backend, const char* packagePath, const char* interfaceProvider, const char* target) {
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorBackend_ValidateOpConfig(Qnn_BackendHandle_t backend, Qnn_OpConfig_t opConfig) {
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorBackend_Free(Qnn_BackendHandle_t backend) {
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorDevice_getInfrastructure(const QnnDevice_Infrastructure_t* deviceInfra) {
    QnnDevice_Infrastructure_t* mutablePtr = const_cast<QnnDevice_Infrastructure_t*>(deviceInfra);
    *mutablePtr = &gQnnConvertorDeviceInfra;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorDevice_Create(Qnn_LogHandle_t logger, const QnnDevice_Config_t** config, Qnn_DeviceHandle_t* device) {
    *device = NOTNULL;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorDevice_Free(Qnn_DeviceHandle_t device) {
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorGraph_Create(Qnn_ContextHandle_t contextHandle, const char* graphName, const QnnGraph_Config_t** config, Qnn_GraphHandle_t* graphHandle) {
    *graphHandle = NOTNULL;

    QNNConvertor::RecordBegin(graphName);

    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorGraph_Finalize(Qnn_GraphHandle_t graphHandle, Qnn_ProfileHandle_t profileHandle, Qnn_SignalHandle_t signalHandle) {
    QNNConvertor::RecordEnd();

    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorGraph_Execute(Qnn_GraphHandle_t graphHandle, const Qnn_Tensor_t* inputs, uint32_t numInputs, Qnn_Tensor_t* outputs, uint32_t numOutputs, Qnn_ProfileHandle_t profileHandle, Qnn_SignalHandle_t signalHandle) {
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorGraph_AddNode(Qnn_GraphHandle_t graphHandle, Qnn_OpConfig_t opConfig) {
    QNNConvertor::RecordNode(opConfig);
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorContext_Create(Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device, const QnnContext_Config_t** config, Qnn_ContextHandle_t* context) {
    *context = NOTNULL;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorContext_Free(Qnn_ContextHandle_t context, Qnn_ProfileHandle_t profile) {
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorTensor_CreateGraphTensor(Qnn_GraphHandle_t graph, Qnn_Tensor_t* tensor) {
    QNNConvertor::RecordTensor(tensor);

    return QNN_SUCCESS;
}

QNN_INTERFACE_VER_TYPE gQnnConvertorInterface = {
    /* propertyHasCapability */                  QnnConvertorProperty_hasCapability,

    /* backendCreate */                          QnnConvertorBackend_Create,
    /* backendSetConfig */                       NULL,
    /* backendGetApiVersion */                   NULL,
    /* backendGetBuildId */                      NULL,
    /* backendRegisterOpPackage */               QnnConvertorBackend_RegisterOpPackage,
    /* backendGetSupportedOperations */          NULL,
    /* backendValidateOpConfig */                QnnConvertorBackend_ValidateOpConfig,
    /* backendFree */                            QnnConvertorBackend_Free,

    /* contextCreate */                          QnnConvertorContext_Create,
    /* contextSetConfig */                       NULL,
    /* contextGetBinarySize */                   NULL,
    /* contextGetBinary */                       NULL,
    /* contextCreateFromBinary */                NULL,
    /* contextFree */                            QnnConvertorContext_Free,

    /* graphCreate */                            QnnConvertorGraph_Create,
    /* graphCreateSubgraph */                    NULL,
    /* graphSetConfig */                         NULL,
    /* graphAddNode */                           QnnConvertorGraph_AddNode,
    /* graphFinalize */                          QnnConvertorGraph_Finalize,
    /* graphRetrieve */                          NULL,
    /* graphExecute */                           QnnConvertorGraph_Execute,
    /* graphExecuteAsync */                      NULL,

    /* tensorCreateContextTensor */              NULL,
    /* tensorCreateGraphTensor */                QnnConvertorTensor_CreateGraphTensor,

    /* logCreate */                              QnnConvertorLog_Create,
    /* logSetLogLevel */                         NULL,
    /* logFree */                                QnnConvertorLog_Free,

    /* profileCreate */                          NULL,
    /* profileSetConfig */                       NULL,
    /* profileGetEvents */                       NULL,
    /* profileGetSubEvents */                    NULL,
    /* profileGetEventData */                    NULL,
    /* profileGetExtendedEventData */            NULL,
    /* profileFree */                            NULL,

    /* memRegister */                            NULL,
    /* memDeRegister */                          NULL,

    /* deviceGetPlatformInfo */                  NULL,
    /* deviceFreePlatformInfo */                 NULL,
    /* deviceGetInfrastructure */                QnnConvertorDevice_getInfrastructure,
    /* deviceCreate */                           QnnConvertorDevice_Create,
    /* deviceSetConfig */                        NULL,
    /* deviceGetInfo */                          NULL,
    /* deviceFree */                             QnnConvertorDevice_Free,

    /* signalCreate */                           NULL,
    /* signalSetConfig */                        NULL,
    /* signalTrigger */                          NULL,
    /* signalFree */                             NULL,

    /* errorGetMessage */                        NULL,
    /* errorGetVerboseMessage */                 NULL,
    /* errorFreeVerboseMessage */                NULL,

    /* graphPrepareExecutionEnvironment */       NULL,
    /* graphReleaseExecutionEnvironment */       NULL,
    /* graphGetProperty */                       NULL,

    /* contextValidateBinary */                  NULL,
    /* contextCreateFromBinaryWithSignal */      NULL,
    /* contextCreateFromBinaryListAsync */       NULL,
    /* tensorUpdateGraphTensors */               NULL,
    /* tensorUpdateContextTensors */             NULL,
    /* contextGetBinarySectionSize */            NULL,
    /* contextGetBinarySection */                NULL,
    /* contextApplyBinarySection */              NULL,
    /* backendGetProperty */                     NULL,
    /* contextGetProperty */                     NULL
};

Qnn_ErrorHandle_t QnnConvertorHtpPerfInfrastructure_SetPowerConfig(uint32_t powerConfigId, const QnnHtpPerfInfrastructure_PowerConfig_t** config) {
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorHtpPerfInfrastructure_CreatePowerConfigId (uint32_t deviceId, uint32_t coreId, uint32_t* powerConfigId) {
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnConvertorHtpPerfInfrastructure_DestroyPowerConfigId (uint32_t powerConfigId) {
    return QNN_SUCCESS;
}

QnnHtpDevice_PerfInfrastructure_t gQnnConvertorPerfInfra = {
    QnnConvertorHtpPerfInfrastructure_CreatePowerConfigId,
    QnnConvertorHtpPerfInfrastructure_DestroyPowerConfigId,
    QnnConvertorHtpPerfInfrastructure_SetPowerConfig,
    NULL
};

QnnHtpDevice_Infrastructure_t gQnnConvertorDeviceInfra = {
    QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF,
    gQnnConvertorPerfInfra
};

Qnn_ErrorHandle_t QnnConvertorSystemContext_create(QnnSystemContext_Handle_t* sysCtxHandle) {
    *sysCtxHandle = NOTNULL;
    return QNN_SUCCESS;
}
Qnn_ErrorHandle_t QnnConvertorSystemContext_getBinaryInfo(QnnSystemContext_Handle_t sysCtxHandle, void* binaryBuffer, uint64_t binaryBufferSize, const QnnSystemContext_BinaryInfo_t** binaryInfo, Qnn_ContextBinarySize_t* binaryInfoSize) {
    return QNN_SUCCESS;
}
Qnn_ErrorHandle_t QnnConvertorSystemContext_free(QnnSystemContext_Handle_t sysCtxHandle) {
    return QNN_SUCCESS;
}
QNN_SYSTEM_INTERFACE_VER_TYPE gQnnConvertorSystemInterface = {
    /*systemContextCreate*/              QnnConvertorSystemContext_create,
    /*systemContextGetBinaryInfo*/       QnnConvertorSystemContext_getBinaryInfo,
    /*systemContextGetMetaData*/         NULL,
    /*systemContextFree*/                QnnConvertorSystemContext_free,
    /*systemTensorGetMemoryFootprint*/   NULL,
};
#endif
} // end namespace MNN
} // end namespace QNN
