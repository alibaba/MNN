//
//  NeuronAdapterSymbol.hpp
//  MNN
//
//  Created by MNN on 2022/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NeuronAdapterSymbols_h
#define NeuronAdapterSymbols_h
#include "NeuronAdapterDefine.hpp"

namespace MNN {
// typedef the function in NeuronAdapter will be used
typedef int (MNN_NeuronModel_getSupportedOperationsForDevices)(const NeuronModel* model, const NeuronDevice* const* devices, uint32_t numDevices, bool* supportedOps);
typedef int (MNN_Neuron_getDeviceCount)(uint32_t* numDevices);
typedef int (MNN_Neuron_getDevice)(uint32_t devIndex, NeuronDevice** device);
typedef int (MNN_NeuronDevice_getName)(const NeuronDevice* device, const char** name);
typedef int (MNN_NeuronCompilation_createForDevices)(NeuronModel* model, const NeuronDevice* const* devices, uint32_t numDevices, NeuronCompilation** compilation);
typedef int (MNN_NeuronExecution_compute)(NeuronExecution* execution);
typedef int (MNN_NeuronModel_create)(NeuronModel** model);
typedef void (MNN_NeuronModel_free)(NeuronModel* model);
typedef int (MNN_NeuronModel_finish)(NeuronModel* model);
typedef int (MNN_NeuronModel_addOperand)(NeuronModel* model, const NeuronOperandType* type);
typedef int (MNN_NeuronModel_setOperandValue)(NeuronModel* model, int32_t index, const void* buffer, size_t length);
typedef int (MNN_NeuronModel_setOperandSymmPerChannelQuantParams)(NeuronModel* model, int32_t index, const NeuronSymmPerChannelQuantParams* channelQuant);
typedef int (MNN_NeuronModel_addOperation)(NeuronModel* model, NeuronOperationType type, uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount, const uint32_t* outputs);
typedef int (MNN_NeuronModel_identifyInputsAndOutputs)(NeuronModel* model, uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount, const uint32_t* outputs);
typedef int (MNN_NeuronCompilation_create)(NeuronModel* model, NeuronCompilation** compilation);
typedef void (MNN_NeuronCompilation_free)(NeuronCompilation* compilation);
typedef int (MNN_NeuronCompilation_setPreference)(NeuronCompilation* compilation, int32_t preference);
typedef int (MNN_NeuronCompilation_finish)(NeuronCompilation* compilation);
typedef int (MNN_NeuronExecution_create)(NeuronCompilation* compilation, NeuronExecution** execution);
typedef void (MNN_NeuronExecution_free)(NeuronExecution* execution);
typedef int (MNN_NeuronExecution_setInput)(NeuronExecution* execution, int32_t index, const NeuronOperandType* type, const void* buffer, size_t length);
typedef int (MNN_NeuronExecution_setInputFromMemory)(NeuronExecution* execution, int32_t index, const NeuronOperandType* type, const NeuronMemory* memory, size_t offset, size_t length);
typedef int (MNN_NeuronExecution_setOutput)(NeuronExecution* execution, int32_t index, const NeuronOperandType* type, void* buffer, size_t length);
typedef int (MNN_NeuronExecution_setOutputFromMemory)(NeuronExecution* execution, int32_t index, const NeuronOperandType* type, const NeuronMemory* memory, size_t offset, size_t length);
typedef int (MNN_NeuronExecution_compute)(NeuronExecution* execution);
typedef int (MNN_NeuronEvent_wait)(NeuronEvent* event);
typedef void (MNN_NeuronEvent_free)(NeuronEvent* event);
typedef int (MNN_NeuronMemory_createFromAHardwareBuffer)(const AHardwareBuffer* ahwb, NeuronMemory** memory);
typedef int (MNN_NeuronMemory_createFromFd)(size_t size, int protect, int fd, size_t offset, NeuronMemory **memory);
typedef void (MNN_NeuronMemory_free)(NeuronMemory* memory);

// symbols
bool loadNeuronAdapterSymbol();
extern MNN_NeuronModel_getSupportedOperationsForDevices *NeuronModel_getSupportedOperationsForDevices_29;
extern MNN_Neuron_getDeviceCount *Neuron_getDeviceCount_29;
extern MNN_Neuron_getDevice *Neuron_getDevice_29;
extern MNN_NeuronDevice_getName *NeuronDevice_getName_29;
extern MNN_NeuronCompilation_createForDevices *NeuronCompilation_createForDevices_29;
extern MNN_NeuronExecution_compute *NeuronExecution_compute_29;
extern MNN_NeuronModel_create *NeuronModel_create_27;
extern MNN_NeuronModel_free *NeuronModel_free_27;
extern MNN_NeuronModel_finish *NeuronModel_finish_27;
extern MNN_NeuronModel_addOperand *NeuronModel_addOperand_27;
extern MNN_NeuronModel_setOperandValue *NeuronModel_setOperandValue_27;
extern MNN_NeuronModel_setOperandSymmPerChannelQuantParams *NeuronModel_setOperandSymmPerChannelQuantParams_29;
extern MNN_NeuronModel_addOperation *NeuronModel_addOperation_27;
extern MNN_NeuronModel_identifyInputsAndOutputs *NeuronModel_identifyInputsAndOutputs_27;
extern MNN_NeuronCompilation_create *NeuronCompilation_create_27;
extern MNN_NeuronCompilation_free *NeuronCompilation_free_27;
extern MNN_NeuronCompilation_setPreference *NeuronCompilation_setPreference_27;
extern MNN_NeuronCompilation_finish *NeuronCompilation_finish_27;
extern MNN_NeuronExecution_create *NeuronExecution_create_27;
extern MNN_NeuronExecution_free *NeuronExecution_free_27;
extern MNN_NeuronExecution_setInput *NeuronExecution_setInput_27;
extern MNN_NeuronExecution_setInputFromMemory *NeuronExecution_setInputFromMemory_27;
extern MNN_NeuronExecution_setOutput *NeuronExecution_setOutput_27;
extern MNN_NeuronExecution_setOutputFromMemory *NeuronExecution_setOutputFromMemory_27;
extern MNN_NeuronExecution_compute *NNeuronExecution_compute_27;
extern MNN_NeuronEvent_wait *NeuronEvent_wait_27;
extern MNN_NeuronEvent_free *NeuronEvent_free_27;
extern MNN_NeuronMemory_createFromAHardwareBuffer *NeuronMemory_createFromAHardwareBuffer_29;
extern MNN_NeuronMemory_createFromFd *NeuronMemory_createFromFd_27;
extern MNN_NeuronMemory_free *NeuronMemory_free_27;
}
#endif /* NeuronAdapterSymbols_h */