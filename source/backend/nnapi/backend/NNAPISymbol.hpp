//
//  NNAPISymbol.hpp
//  MNN
//
//  Created by MNN on 2022/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NNAPISymbols_h
#define NNAPISymbols_h
#include "NNAPIDefine.hpp"

namespace MNN {
// typedef the function in nnapi will be used
typedef int (MNN_ANeuralNetworksModel_getSupportedOperationsForDevices)(const ANeuralNetworksModel* model, const ANeuralNetworksDevice* const* devices, uint32_t numDevices, bool* supportedOps);
typedef int (MNN_ANeuralNetworks_getDeviceCount)(uint32_t* numDevices);
typedef int (MNN_ANeuralNetworks_getDevice)(uint32_t devIndex, ANeuralNetworksDevice** device);
typedef int (MNN_ANeuralNetworksDevice_getName)(const ANeuralNetworksDevice* device, const char** name);
typedef int (MNN_ANeuralNetworksDevice_getType)(const ANeuralNetworksDevice* device, int32_t* type);
typedef int (MNN_ANeuralNetworksCompilation_createForDevices)(ANeuralNetworksModel* model, const ANeuralNetworksDevice* const* devices, uint32_t numDevices, ANeuralNetworksCompilation** compilation);
typedef int (MNN_ANeuralNetworksExecution_compute)(ANeuralNetworksExecution* execution);
typedef int (MNN_ANeuralNetworksBurst_create)(ANeuralNetworksCompilation* compilation, ANeuralNetworksBurst** burst);
typedef void (MNN_ANeuralNetworksBurst_free)(ANeuralNetworksBurst* burst);
typedef int (MNN_ANeuralNetworksExecution_burstCompute)(ANeuralNetworksExecution* execution, ANeuralNetworksBurst* burst);
typedef int (MNN_ANeuralNetworksModel_create)(ANeuralNetworksModel** model);
typedef void (MNN_ANeuralNetworksModel_free)(ANeuralNetworksModel* model);
typedef int (MNN_ANeuralNetworksModel_finish)(ANeuralNetworksModel* model);
typedef int (MNN_ANeuralNetworksModel_addOperand)(ANeuralNetworksModel* model, const ANeuralNetworksOperandType* type);
typedef int (MNN_ANeuralNetworksModel_setOperandValue)(ANeuralNetworksModel* model, int32_t index, const void* buffer, size_t length);
typedef int (MNN_ANeuralNetworksModel_setOperandSymmPerChannelQuantParams)(ANeuralNetworksModel* model, int32_t index, const ANeuralNetworksSymmPerChannelQuantParams* channelQuant);
typedef int (MNN_ANeuralNetworksModel_addOperation)(ANeuralNetworksModel* model, ANeuralNetworksOperationType type, uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount, const uint32_t* outputs);
typedef int (MNN_ANeuralNetworksModel_identifyInputsAndOutputs)(ANeuralNetworksModel* model, uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount, const uint32_t* outputs);
typedef int (MNN_ANeuralNetworksCompilation_create)(ANeuralNetworksModel* model, ANeuralNetworksCompilation** compilation);
typedef void (MNN_ANeuralNetworksCompilation_free)(ANeuralNetworksCompilation* compilation);
typedef int (MNN_ANeuralNetworksCompilation_setPreference)(ANeuralNetworksCompilation* compilation, int32_t preference);
typedef int (MNN_ANeuralNetworksCompilation_finish)(ANeuralNetworksCompilation* compilation);
typedef int (MNN_ANeuralNetworksExecution_create)(ANeuralNetworksCompilation* compilation, ANeuralNetworksExecution** execution);
typedef void (MNN_ANeuralNetworksExecution_free)(ANeuralNetworksExecution* execution);
typedef int (MNN_ANeuralNetworksExecution_setInput)(ANeuralNetworksExecution* execution, int32_t index, const ANeuralNetworksOperandType* type, const void* buffer, size_t length);
typedef int (MNN_ANeuralNetworksExecution_setInputFromMemory)(ANeuralNetworksExecution* execution, int32_t index, const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory, size_t offset, size_t length);
typedef int (MNN_ANeuralNetworksExecution_setOutput)(ANeuralNetworksExecution* execution, int32_t index, const ANeuralNetworksOperandType* type, void* buffer, size_t length);
typedef int (MNN_ANeuralNetworksExecution_setOutputFromMemory)(ANeuralNetworksExecution* execution, int32_t index, const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory, size_t offset, size_t length);
typedef int (MNN_ANeuralNetworksExecution_startCompute)(ANeuralNetworksExecution* execution, ANeuralNetworksEvent** event);
typedef int (MNN_ANeuralNetworksEvent_wait)(ANeuralNetworksEvent* event);
typedef void (MNN_ANeuralNetworksEvent_free)(ANeuralNetworksEvent* event);
typedef int (MNN_ANeuralNetworksDevice_getVersion)(const ANeuralNetworksDevice* device, const char** version);
typedef int (MNN_ANeuralNetworksMemory_createFromAHardwareBuffer)(const AHardwareBuffer* ahwb, ANeuralNetworksMemory** memory);
typedef int (MNN_ANeuralNetworksMemory_createFromFd)(size_t size, int protect, int fd, size_t offset, ANeuralNetworksMemory **memory);
typedef void (MNN_ANeuralNetworksMemory_free)(ANeuralNetworksMemory* memory);
typedef void (MNN_ANeuralNetworksExecution_setMeasureTiming)(ANeuralNetworksExecution* execution, bool measure);
typedef void (MNN_ANeuralNetworksExecution_getDuration)(const ANeuralNetworksExecution* execution,int32_t durationCode, uint64_t* duration);

// symbols
bool loadNNAPISymbol();
extern MNN_ANeuralNetworksModel_getSupportedOperationsForDevices *ANeuralNetworksModel_getSupportedOperationsForDevices_29;
extern MNN_ANeuralNetworks_getDeviceCount *ANeuralNetworks_getDeviceCount_29;
extern MNN_ANeuralNetworks_getDevice *ANeuralNetworks_getDevice_29;
extern MNN_ANeuralNetworksDevice_getName *ANeuralNetworksDevice_getName_29;
extern MNN_ANeuralNetworksDevice_getType *ANeuralNetworksDevice_getType_29;
extern MNN_ANeuralNetworksCompilation_createForDevices *ANeuralNetworksCompilation_createForDevices_29;
extern MNN_ANeuralNetworksExecution_compute *ANeuralNetworksExecution_compute_29;
extern MNN_ANeuralNetworksBurst_create *ANeuralNetworksBurst_create_29;
extern MNN_ANeuralNetworksBurst_free *ANeuralNetworksBurst_free_29;
extern MNN_ANeuralNetworksExecution_burstCompute *ANeuralNetworksExecution_burstCompute_29;
extern MNN_ANeuralNetworksModel_create *ANeuralNetworksModel_create_27;
extern MNN_ANeuralNetworksModel_free *ANeuralNetworksModel_free_27;
extern MNN_ANeuralNetworksModel_finish *ANeuralNetworksModel_finish_27;
extern MNN_ANeuralNetworksModel_addOperand *ANeuralNetworksModel_addOperand_27;
extern MNN_ANeuralNetworksModel_setOperandValue *ANeuralNetworksModel_setOperandValue_27;
extern MNN_ANeuralNetworksModel_setOperandSymmPerChannelQuantParams *ANeuralNetworksModel_setOperandSymmPerChannelQuantParams_29;
extern MNN_ANeuralNetworksModel_addOperation *ANeuralNetworksModel_addOperation_27;
extern MNN_ANeuralNetworksModel_identifyInputsAndOutputs *ANeuralNetworksModel_identifyInputsAndOutputs_27;
extern MNN_ANeuralNetworksCompilation_create *ANeuralNetworksCompilation_create_27;
extern MNN_ANeuralNetworksCompilation_free *ANeuralNetworksCompilation_free_27;
extern MNN_ANeuralNetworksCompilation_setPreference *ANeuralNetworksCompilation_setPreference_27;
extern MNN_ANeuralNetworksCompilation_finish *ANeuralNetworksCompilation_finish_27;
extern MNN_ANeuralNetworksExecution_create *ANeuralNetworksExecution_create_27;
extern MNN_ANeuralNetworksExecution_free *ANeuralNetworksExecution_free_27;
extern MNN_ANeuralNetworksExecution_setInput *ANeuralNetworksExecution_setInput_27;
extern MNN_ANeuralNetworksExecution_setInputFromMemory *ANeuralNetworksExecution_setInputFromMemory_27;
extern MNN_ANeuralNetworksExecution_setOutput *ANeuralNetworksExecution_setOutput_27;
extern MNN_ANeuralNetworksExecution_setOutputFromMemory *ANeuralNetworksExecution_setOutputFromMemory_27;
extern MNN_ANeuralNetworksExecution_startCompute *ANeuralNetworksExecution_startCompute_27;
extern MNN_ANeuralNetworksEvent_wait *ANeuralNetworksEvent_wait_27;
extern MNN_ANeuralNetworksEvent_free *ANeuralNetworksEvent_free_27;
extern MNN_ANeuralNetworksDevice_getVersion *ANeuralNetworksDevice_getVersion_29;
extern MNN_ANeuralNetworksMemory_createFromAHardwareBuffer *ANeuralNetworksMemory_createFromAHardwareBuffer_29;
extern MNN_ANeuralNetworksMemory_createFromFd *ANeuralNetworksMemory_createFromFd_27;
extern MNN_ANeuralNetworksMemory_free *ANeuralNetworksMemory_free_27;
extern MNN_ANeuralNetworksExecution_setMeasureTiming *ANeuralNetworksExecution_setMeasureTiming_29;
extern MNN_ANeuralNetworksExecution_getDuration *ANeuralNetworksExecution_getDuration_29;
}
#endif /* NNAPISymbols_h */