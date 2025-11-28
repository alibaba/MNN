/* Copyright Statement:
 *
 * This software/firmware and related documentation ("MediaTek Software") are
 * protected under relevant copyright laws. The information contained herein
 * is confidential and proprietary to MediaTek Inc. and/or its licensors.
 * Without the prior written permission of MediaTek inc. and/or its licensors,
 * any reproduction, modification, use or disclosure of MediaTek Software,
 * and information contained herein, in whole or in part, shall be strictly prohibited.
 */
/* MediaTek Inc. (C) 2020. All rights reserved.
 *
 * BY OPENING THIS FILE, RECEIVER HEREBY UNEQUIVOCALLY ACKNOWLEDGES AND AGREES
 * THAT THE SOFTWARE/FIRMWARE AND ITS DOCUMENTATIONS ("MEDIATEK SOFTWARE")
 * RECEIVED FROM MEDIATEK AND/OR ITS REPRESENTATIVES ARE PROVIDED TO RECEIVER ON
 * AN "AS-IS" BASIS ONLY. MEDIATEK EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE OR NONINFRINGEMENT.
 * NEITHER DOES MEDIATEK PROVIDE ANY WARRANTY WHATSOEVER WITH RESPECT TO THE
 * SOFTWARE OF ANY THIRD PARTY WHICH MAY BE USED BY, INCORPORATED IN, OR
 * SUPPLIED WITH THE MEDIATEK SOFTWARE, AND RECEIVER AGREES TO LOOK ONLY TO SUCH
 * THIRD PARTY FOR ANY WARRANTY CLAIM RELATING THERETO. RECEIVER EXPRESSLY ACKNOWLEDGES
 * THAT IT IS RECEIVER'S SOLE RESPONSIBILITY TO OBTAIN FROM ANY THIRD PARTY ALL PROPER LICENSES
 * CONTAINED IN MEDIATEK SOFTWARE. MEDIATEK SHALL ALSO NOT BE RESPONSIBLE FOR ANY MEDIATEK
 * SOFTWARE RELEASES MADE TO RECEIVER'S SPECIFICATION OR TO CONFORM TO A PARTICULAR
 * STANDARD OR OPEN FORUM. RECEIVER'S SOLE AND EXCLUSIVE REMEDY AND MEDIATEK'S ENTIRE AND
 * CUMULATIVE LIABILITY WITH RESPECT TO THE MEDIATEK SOFTWARE RELEASED HEREUNDER WILL BE,
 * AT MEDIATEK'S OPTION, TO REVISE OR REPLACE THE MEDIATEK SOFTWARE AT ISSUE,
 * OR REFUND ANY SOFTWARE LICENSE FEES OR SERVICE CHARGE PAID BY RECEIVER TO
 * MEDIATEK FOR SUCH MEDIATEK SOFTWARE AT ISSUE.
 *
 * The following software/firmware and/or related documentation ("MediaTek Software")
 * have been modified by MediaTek Inc. All revisions are subject to any receiver's
 * applicable license agreements with MediaTek Inc.
 */

#pragma once

#include <android/log.h>
#include <dlfcn.h>
#include "NeuronAdapter.h"

#define LOAD_ADAPTER_FUNCTION(name) \
    static name##_fn fn = reinterpret_cast<name##_fn>(loadAdapterFunction(#name));

#define EXECUTE_ADAPTER_FUNCTION(...) \
    if (fn != nullptr) {              \
        fn(__VA_ARGS__);              \
    }

#define EXECUTE_ADAPTER_FUNCTION_RETURN_INT(...) return fn != nullptr ? fn(__VA_ARGS__) : -1;

#define EXECUTE_ADAPTER_FUNCTION_RETURN_BOOL(...) return fn != nullptr ? fn(__VA_ARGS__) : false;

#define NEURON_ADAPTER_SHIM_TAG "NeuronAdapterShim"

static void* sHandle = nullptr;
inline void* loadAdapterLibrary(const char* name) {
    sHandle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (sHandle == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, NEURON_ADAPTER_SHIM_TAG, "Unable to open library %s", name);
    }
    return sHandle;
}

inline void* getAdapterLibraryHandle() {
    if (sHandle == nullptr) {
        sHandle = loadAdapterLibrary("libneuronusdk_adapter.mtk.so");
    }
    if (sHandle == nullptr) {
        sHandle = loadAdapterLibrary("libneuron_adapter_mgvi.so");
    }
    if (sHandle == nullptr) {
        sHandle = loadAdapterLibrary("libneuron_adapter.so");
    }
    return sHandle;
}

inline void* loadAdapterFunction(const char* name) {
    void* fn = nullptr;
    if (getAdapterLibraryHandle() != nullptr) {
        fn = dlsym(getAdapterLibraryHandle(), name);
    }

    if (fn == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, NEURON_ADAPTER_SHIM_TAG, "Unable to open function %s", name);
    }

    return fn;
}

/*************************************************************************************************/
typedef int (*Neuron_getVersion_fn)(NeuronRuntimeVersion* version);

typedef int (*Neuron_getFeatureSupportedStatus_fn)(NeuronFeatureType type, bool* supported);

typedef int (*Neuron_getNeuroPilotMagicNumber_fn)(int32_t* magic);

typedef int (*Neuron_getL1MemorySizeKb_fn)(uint32_t* sizeKb);

typedef int (*NeuronModel_create_fn)(NeuronModel** model);

typedef void (*NeuronModel_free_fn)(NeuronModel* model);

typedef int (*NeuronModel_finish_fn)(NeuronModel* model);

typedef int (*NeuronModel_addOperand_fn)(NeuronModel* model, const NeuronOperandType* type);

typedef int (*NeuronModel_setOperandValue_fn)(NeuronModel* model, int32_t index, const void* buffer,
                                              size_t length);

typedef int (*NeuronModel_setOperandValueFromModel_fn)(NeuronModel* model, int32_t index,
                                                       const NeuronModel* value);

typedef int (*NeuronModel_setOperandSymmPerChannelQuantParams_fn)(
    NeuronModel* model, int32_t index, const NeuronSymmPerChannelQuantParams* channelQuant);

typedef int (*NeuronModel_setOperandPerChannelQuantParams_fn)(
    NeuronModel* model, int32_t index, const NeuronPerChannelQuantParams* channelQuant);

typedef int (*NeuronModel_addOperation_fn)(NeuronModel* model, NeuronOperationType type,
                                           uint32_t inputCount, const uint32_t* inputs,
                                           uint32_t outputCount, const uint32_t* outputs);

typedef int (*NeuronModel_addOperationExtension_fn)(NeuronModel* model, const char* name,
                                                    const char* vendor, const NeuronDevice* device,
                                                    uint32_t inputCount, const uint32_t* inputs,
                                                    uint32_t outputCount, const uint32_t* outputs);

typedef int (*NeuronModel_identifyInputsAndOutputs_fn)(NeuronModel* model, uint32_t inputCount,
                                                       const uint32_t* inputs, uint32_t outputCount,
                                                       const uint32_t* outputs);

typedef int (*NeuronModel_getSupportedOperations_fn)(NeuronModel* model, bool* supported,
                                                     uint32_t operationCount);

typedef int (*NeuronModel_getSupportedOperationsForDevices_fn)(const NeuronModel* model,
                                                               const NeuronDevice* const* devices,
                                                               uint32_t numDevices,
                                                               bool* supportedOps);

typedef int (*NeuronModel_relaxComputationFloat32toFloat16_fn)(NeuronModel* model, bool allow);

typedef int (*NeuronModel_suppressInputConversion_fn)(NeuronModel* model, bool suppress);

typedef int (*NeuronModel_suppressOutputConversion_fn)(NeuronModel* model, bool suppress);

typedef int (*NeuronModel_restoreFromCompiledNetwork_fn)(NeuronModel** model,
                                                         NeuronCompilation** compilation,
                                                         const void* buffer, const size_t size);

typedef int (*NeuronCompilation_create_fn)(NeuronModel* model, NeuronCompilation** compilation);

typedef int (*NeuronCompilation_createForDevices_fn)(NeuronModel* model,
                                                     const NeuronDevice* const* devices,
                                                     uint32_t numDevices,
                                                     NeuronCompilation** compilation);

typedef int (*NeuronCompilation_createForDebug_fn)(NeuronModel* model,
                                                   NeuronCompilation** compilation);

typedef void (*NeuronCompilation_free_fn)(NeuronCompilation* compilation);

typedef int (*NeuronCompilation_finish_fn)(NeuronCompilation* compilation);

typedef int (*NeuronCompilation_setCaching_fn)(NeuronCompilation* compilation, const char* cacheDir,
                                               const uint8_t* token);

typedef int (*NeuronCompilation_setPreference_fn)(NeuronCompilation* compilation,
                                                  int32_t preference);

typedef int (*NeuronCompilation_setPriority_fn)(NeuronCompilation* compilation, int32_t priority);

typedef int (*NeuronCompilation_getInputPaddedDimensions_fn)(NeuronCompilation* compilation,
                                                             int32_t index, uint32_t* dimensions);

typedef int (*NeuronCompilation_getOutputPaddedDimensions_fn)(NeuronCompilation* compilation,
                                                              int32_t index, uint32_t* dimensions);

typedef int (*NeuronCompilation_getInputPaddedSize_fn)(NeuronCompilation* compilation,
                                                       int32_t index, size_t* size);

typedef int (*NeuronCompilation_getOutputPaddedSize_fn)(NeuronCompilation* compilation,
                                                        int32_t index, size_t* size);

typedef int (*NeuronCompilation_getCompiledNetworkSize_fn)(NeuronCompilation* compilation,
                                                           size_t* size);

typedef int (*NeuronCompilation_storeCompiledNetwork_fn)(NeuronCompilation* compilation,
                                                         void* buffer, const size_t size);

typedef int (*NeuronCompilation_setOptimizationHint_fn)(NeuronCompilation* compilation,
                                                        uint32_t optimizationCode);

typedef int (*NeuronCompilation_setOptimizationString_fn)(NeuronCompilation* compilation,
                                                          const char* optimizationString);

typedef int (*NeuronCompilation_setTrimIOAlignment_fn)(NeuronCompilation* compilation, bool enable);

typedef int (*NeuronCompilation_setSWDilatedConv_fn)(NeuronCompilation* compilation, bool enable);

typedef int (*NeuronExecution_create_fn)(NeuronCompilation* compilation,
                                         NeuronExecution** execution);

typedef void (*NeuronExecution_free_fn)(NeuronExecution* execution);

typedef int (*NeuronExecution_setInput_fn)(NeuronExecution* execution, int32_t index,
                                           const NeuronOperandType* type, const void* buffer,
                                           size_t length);

typedef int (*NeuronExecution_setOutput_fn)(NeuronExecution* execution, int32_t index,
                                            const NeuronOperandType* type, void* buffer,
                                            size_t length);

typedef int (*NeuronExecution_setInputFromMemory_fn)(NeuronExecution* execution, uint32_t index,
                                                     const NeuronOperandType* type,
                                                     const NeuronMemory* memory, size_t offset,
                                                     size_t length);

typedef int (*NeuronExecution_setOutputFromMemory_fn)(NeuronExecution* execution, uint32_t index,
                                                      const NeuronOperandType* type,
                                                      const NeuronMemory* memory, size_t offset,
                                                      size_t length);

typedef int (*NeuronMemory_createFromFd_fn)(size_t size, int protect, int fd, size_t offset,
                                            NeuronMemory** memory);

typedef int (*NeuronMemory_createFromAHardwareBuffer_fn)(const AHardwareBuffer* ahwb,
                                                         NeuronMemory** memory);

typedef void (*NeuronMemory_free_fn)(NeuronMemory* memory);

typedef int (*NeuronExecution_compute_fn)(NeuronExecution* execution);

typedef int (*NeuronExecution_startComputeWithDependencies_fn)(
    NeuronExecution* execution, const NeuronEvent* const* dependencies, uint32_t num_dependencies,
    uint64_t duration, NeuronEvent** event);

typedef int (*NeuronEvent_getSyncFenceFd_fn)(const NeuronEvent* event, int* syncFenceFd);

typedef int (*NeuronEvent_wait_fn)(NeuronEvent* event);

typedef void (*NeuronEvent_free_fn)(NeuronEvent* event);

typedef int (*NeuronExecution_setLoopTimeout_fn)(NeuronExecution* execution, uint64_t duration);

typedef int (*NeuronExecution_setBoostHint_fn)(NeuronExecution* execution, uint8_t boostValue);

typedef int (*NeuronCompilation_createForMultiExecutions_fn)(NeuronModel* model,
                                                             NeuronCompilation** compilation);

typedef int (*NeuronDebug_setReportPath_fn)(NeuronModel* model, const char* path);

typedef int (*Neuron_getDeviceCount_fn)(uint32_t* numDevices);

typedef int (*Neuron_getDevice_fn)(uint32_t devIndex, NeuronDevice** device);

typedef int (*NeuronDevice_getName_fn)(const NeuronDevice* device, const char** name);

typedef int (*NeuronDevice_getDescription_fn)(const NeuronDevice* device, const char** description);

typedef int (*NeuronDevice_getExtensionSupport_fn)(const char* extensionName,
                                                   bool* isExtensionSupported);

typedef int (*NeuronModel_getExtensionOperandType_fn)(NeuronModel* model, const char* extensionName,
                                                      uint16_t operandCodeWithinExtension,
                                                      int32_t* type);

typedef int (*NeuronModel_getExtensionOperationType_fn)(NeuronModel* model,
                                                        const char* extensionName,
                                                        uint16_t operationCodeWithinExtension,
                                                        int32_t* type);

typedef int (*NeuronModel_setOperandExtensionData_fn)(NeuronModel* model, int32_t index,
                                                      const void* data, size_t length);

typedef int (*NeuronCompilation_createForBatch_fn)(NeuronModel* model,
                                                   NeuronCompilation** compilation);

typedef int (*NeuronModel_restoreFromCompiledNetworkV2_fn)(NeuronModel** model,
                                                           NeuronCompilation** compilation,
                                                           const void* buffer, const size_t size,
                                                           const CompilationType& type);

typedef int (*NeuronExecution_setRunnerPoolSize_fn)(NeuronExecution* execution, uint8_t numRunners);

typedef int (*NeuronExecution_setBatchDone_fn)(NeuronExecution* execution);

typedef int (*NeuronCompilation_createWithOptions_fn)(NeuronModel* model,
                                                      NeuronCompilation** compilation,
                                                      const char* options);

/*************************************************************************************************/

inline int Neuron_getVersion(NeuronRuntimeVersion* version) {
    LOAD_ADAPTER_FUNCTION(Neuron_getVersion);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(version);
}

inline int Neuron_getFeatureSupportedStatus(NeuronFeatureType type, bool* supported) {
    LOAD_ADAPTER_FUNCTION(Neuron_getFeatureSupportedStatus);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(type, supported);
}

inline int Neuron_getNeuroPilotMagicNumber(int32_t* magic) {
    LOAD_ADAPTER_FUNCTION(Neuron_getNeuroPilotMagicNumber);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(magic);
}

inline int Neuron_getL1MemorySizeKb(uint32_t* sizeKb) {
    LOAD_ADAPTER_FUNCTION(Neuron_getL1MemorySizeKb);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(sizeKb);
}

inline int NeuronModel_create(NeuronModel** model) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_create);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model);
}

inline void NeuronModel_free(NeuronModel* model) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_free);
    EXECUTE_ADAPTER_FUNCTION(model);
}

inline int NeuronModel_finish(NeuronModel* model) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_finish);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model);
}

inline int NeuronModel_addOperand(NeuronModel* model, const NeuronOperandType* type) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_addOperand);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, type);
}

inline int NeuronModel_setOperandValue(NeuronModel* model, int32_t index, const void* buffer,
                                       size_t length) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_setOperandValue);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, index, buffer, length);
}

inline int NeuronModel_setOperandValueFromModel(NeuronModel* model, int32_t index,
                                                const NeuronModel* value) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_setOperandValueFromModel);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, index, value);
}

inline int NeuronModel_setOperandSymmPerChannelQuantParams(
    NeuronModel* model, int32_t index, const NeuronSymmPerChannelQuantParams* channelQuant) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_setOperandSymmPerChannelQuantParams);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, index, channelQuant);
}

inline int NeuronModel_setOperandPerChannelQuantParams(
    NeuronModel* model, int32_t index, const NeuronPerChannelQuantParams* channelQuant) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_setOperandPerChannelQuantParams);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, index, channelQuant);
}

inline int NeuronModel_addOperation(NeuronModel* model, NeuronOperationType type,
                                    uint32_t inputCount, const uint32_t* inputs,
                                    uint32_t outputCount, const uint32_t* outputs) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_addOperation);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, type, inputCount, inputs, outputCount, outputs);
}

inline int NeuronModel_addOperationExtension(NeuronModel* model, const char* name,
                                             const char* vendor, const NeuronDevice* device,
                                             uint32_t inputCount, const uint32_t* inputs,
                                             uint32_t outputCount, const uint32_t* outputs) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_addOperationExtension);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, name, vendor, device, inputCount, inputs,
                                        outputCount, outputs);
}

inline int NeuronModel_identifyInputsAndOutputs(NeuronModel* model, uint32_t inputCount,
                                                const uint32_t* inputs, uint32_t outputCount,
                                                const uint32_t* outputs) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_identifyInputsAndOutputs);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, inputCount, inputs, outputCount, outputs);
}

inline int NeuronModel_getSupportedOperations(NeuronModel* model, bool* supported,
                                              uint32_t operationCount) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_getSupportedOperations);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, supported, operationCount);
}

inline int NeuronModel_getSupportedOperationsForDevices(const NeuronModel* model,
                                                        const NeuronDevice* const* devices,
                                                        uint32_t numDevices, bool* supportedOps) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_getSupportedOperationsForDevices);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, devices, numDevices, supportedOps);
}

inline int NeuronModel_relaxComputationFloat32toFloat16(NeuronModel* model, bool allow) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_relaxComputationFloat32toFloat16);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, allow);
}

inline int NeuronModel_suppressInputConversion(NeuronModel* model, bool suppress) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_suppressInputConversion);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, suppress);
}

inline int NeuronModel_suppressOutputConversion(NeuronModel* model, bool suppress) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_suppressOutputConversion);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, suppress);
}

inline int NeuronModel_restoreFromCompiledNetwork(NeuronModel** model,
                                                  NeuronCompilation** compilation,
                                                  const void* buffer, const size_t size) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_restoreFromCompiledNetwork);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, compilation, buffer, size);
}

// inline int NeuronModel_setName(NeuronModel* model, const char* name) {
//     LOAD_ADAPTER_FUNCTION(NeuronModel_setName);
//     EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, name);
// }

inline int NeuronCompilation_create(NeuronModel* model, NeuronCompilation** compilation) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_create);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, compilation);
}

// inline int NeuronCompilation_createV2(NeuronModel* model, CompilationType type, const char* options,
//                                NeuronCompilation** compilation) {
//     LOAD_ADAPTER_FUNCTION(NeuronCompilation_createV2);
//     EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, type, options, compilation);
// }

inline int NeuronCompilation_createForDevices(NeuronModel* model,
                                              const NeuronDevice* const* devices,
                                              uint32_t numDevices,
                                              NeuronCompilation** compilation) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_createForDevices);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, devices, numDevices, compilation);
}

inline int NeuronCompilation_createForDebug(NeuronModel* model, NeuronCompilation** compilation) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_createForDebug);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, compilation);
}

inline void NeuronCompilation_free(NeuronCompilation* compilation) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_free);
    EXECUTE_ADAPTER_FUNCTION(compilation);
}

inline int NeuronCompilation_finish(NeuronCompilation* compilation) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_finish);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(compilation);
}

// int NeuronCompilation_getSupportedOperations(NeuronCompilation* compilation,
//                                              uint32_t operationCount, bool* supported){
//     LOAD_ADAPTER_FUNCTION(NeuronCompilation_getSupportedOperations);
//     EXECUTE_ADAPTER_FUNCTION_RETURN_INT(compilation, operationCount, supported);
// }

inline int NeuronCompilation_setCaching(NeuronCompilation* compilation, const char* cacheDir,
                                        const uint8_t* token) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_setCaching);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(compilation, cacheDir, token);
}

inline int NeuronCompilation_setPreference(NeuronCompilation* compilation, int32_t preference) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_setPreference);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(compilation, preference);
}

inline int NeuronCompilation_setPriority(NeuronCompilation* compilation, int32_t priority) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_setPriority);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(compilation, priority);
}

inline int NeuronCompilation_getInputPaddedDimensions(NeuronCompilation* compilation, int32_t index,
                                                      uint32_t* dimensions) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_getInputPaddedDimensions);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(compilation, index, dimensions);
}

inline int NeuronCompilation_getOutputPaddedDimensions(NeuronCompilation* compilation,
                                                       int32_t index, uint32_t* dimensions) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_getOutputPaddedDimensions);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(compilation, index, dimensions);
}

inline int NeuronCompilation_getInputPaddedSize(NeuronCompilation* compilation, int32_t index,
                                                size_t* size) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_getInputPaddedSize);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(compilation, index, size);
}

inline int NeuronCompilation_getOutputPaddedSize(NeuronCompilation* compilation, int32_t index,
                                                 size_t* size) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_getOutputPaddedSize);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(compilation, index, size);
}

inline int NeuronCompilation_getCompiledNetworkSize(NeuronCompilation* compilation, size_t* size) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_getCompiledNetworkSize);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(compilation, size);
}

inline int NeuronCompilation_storeCompiledNetwork(NeuronCompilation* compilation, void* buffer,
                                                  const size_t size) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_storeCompiledNetwork);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(compilation, buffer, size);
}

inline int NeuronCompilation_setOptimizationHint(NeuronCompilation* compilation,
                                                 uint32_t optimizationCode) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_setOptimizationHint);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(compilation, optimizationCode);
}

inline int NeuronCompilation_setOptimizationString(NeuronCompilation* compilation,
                                                   const char* optimizationString) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_setOptimizationString);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(compilation, optimizationString);
}

inline int NeuronCompilation_setTrimIOAlignment(NeuronCompilation* compilation, bool enable) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_setTrimIOAlignment);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(compilation, enable);
}

inline int NeuronCompilation_setSWDilatedConv(NeuronCompilation* compilation, bool enable) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_setSWDilatedConv);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(compilation, enable);
}

inline int NeuronExecution_create(NeuronCompilation* compilation, NeuronExecution** execution) {
    LOAD_ADAPTER_FUNCTION(NeuronExecution_create);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(compilation, execution);
}

inline void NeuronExecution_free(NeuronExecution* execution) {
    LOAD_ADAPTER_FUNCTION(NeuronExecution_free);
    EXECUTE_ADAPTER_FUNCTION(execution);
}

inline int NeuronExecution_setInput(NeuronExecution* execution, int32_t index,
                                    const NeuronOperandType* type, const void* buffer,
                                    size_t length) {
    LOAD_ADAPTER_FUNCTION(NeuronExecution_setInput);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(execution, index, type, buffer, length);
}

inline int NeuronExecution_setInputFromMemory(NeuronExecution* execution, uint32_t index,
                                              const NeuronOperandType* type,
                                              const NeuronMemory* memory, size_t offset,
                                              size_t length) {
    LOAD_ADAPTER_FUNCTION(NeuronExecution_setInputFromMemory);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(execution, index, type, memory, offset, length);
}

inline int NeuronExecution_setOutput(NeuronExecution* execution, int32_t index,
                                     const NeuronOperandType* type, void* buffer, size_t length) {
    LOAD_ADAPTER_FUNCTION(NeuronExecution_setOutput);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(execution, index, type, buffer, length);
}

inline int NeuronExecution_setOutputFromMemory(NeuronExecution* execution, uint32_t index,
                                               const NeuronOperandType* type,
                                               const NeuronMemory* memory, size_t offset,
                                               size_t length) {
    LOAD_ADAPTER_FUNCTION(NeuronExecution_setOutputFromMemory);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(execution, index, type, memory, offset, length);
}

inline int NeuronMemory_createFromFd(size_t size, int protect, int fd, size_t offset,
                                     NeuronMemory** memory) {
    LOAD_ADAPTER_FUNCTION(NeuronMemory_createFromFd);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(size, protect, fd, offset, memory);
}

inline int NeuronMemory_createFromAHardwareBuffer(const AHardwareBuffer* ahwb,
                                                  NeuronMemory** memory) {
    LOAD_ADAPTER_FUNCTION(NeuronMemory_createFromAHardwareBuffer);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(ahwb, memory);
}

inline void NeuronMemory_free(NeuronMemory* memory) {
    LOAD_ADAPTER_FUNCTION(NeuronMemory_free);
    EXECUTE_ADAPTER_FUNCTION(memory);
}

inline int NeuronExecution_compute(NeuronExecution* execution) {
    LOAD_ADAPTER_FUNCTION(NeuronExecution_compute);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(execution);
}

inline int NeuronExecution_startComputeWithDependencies(NeuronExecution* execution,
                                                        const NeuronEvent* const* dependencies,
                                                        uint32_t num_dependencies,
                                                        uint64_t duration, NeuronEvent** event) {
    LOAD_ADAPTER_FUNCTION(NeuronExecution_startComputeWithDependencies);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(execution, dependencies, num_dependencies, duration, event);
}

inline int NeuronEvent_getSyncFenceFd(const NeuronEvent* event, int* syncFenceFd) {
    LOAD_ADAPTER_FUNCTION(NeuronEvent_getSyncFenceFd);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(event, syncFenceFd);
}

inline int NeuronEvent_wait(NeuronEvent* event) {
    LOAD_ADAPTER_FUNCTION(NeuronEvent_wait);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(event);
}

inline void NeuronEvent_free(NeuronEvent* event) {
    LOAD_ADAPTER_FUNCTION(NeuronEvent_free);
    EXECUTE_ADAPTER_FUNCTION(event);
}

inline int NeuronExecution_setLoopTimeout(NeuronExecution* execution, uint64_t duration) {
    LOAD_ADAPTER_FUNCTION(NeuronExecution_setLoopTimeout);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(execution, duration);
}

inline int NeuronExecution_setBoostHint(NeuronExecution* execution, uint8_t boostValue) {
    LOAD_ADAPTER_FUNCTION(NeuronExecution_setBoostHint);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(execution, boostValue);
}

inline int NeuronCompilation_createForMultiExecutions(NeuronModel* model,
                                                      NeuronCompilation** compilation) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_createForMultiExecutions);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, compilation);
}

inline int NeuronDebug_setReportPath(NeuronModel* model, const char* path) {
    LOAD_ADAPTER_FUNCTION(NeuronDebug_setReportPath);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, path);
}

inline int Neuron_getDeviceCount(uint32_t* numDevices) {
    LOAD_ADAPTER_FUNCTION(Neuron_getDeviceCount);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(numDevices);
}

inline int Neuron_getDevice(uint32_t devIndex, NeuronDevice** device) {
    LOAD_ADAPTER_FUNCTION(Neuron_getDevice);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(devIndex, device);
}

inline int NeuronDevice_getName(const NeuronDevice* device, const char** name) {
    LOAD_ADAPTER_FUNCTION(NeuronDevice_getName);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(device, name);
}

inline int NeuronDevice_getDescription(const NeuronDevice* device, const char** description) {
    LOAD_ADAPTER_FUNCTION(NeuronDevice_getDescription);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(device, description);
}

inline int NeuronDevice_getExtensionSupport(const char* extensionName, bool* isExtensionSupported) {
    LOAD_ADAPTER_FUNCTION(NeuronDevice_getExtensionSupport);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(extensionName, isExtensionSupported);
}

inline int NeuronModel_getExtensionOperandType(NeuronModel* model, const char* extensionName,
                                               uint16_t operandCodeWithinExtension, int32_t* type) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_getExtensionOperandType);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, extensionName, operandCodeWithinExtension, type);
}

inline int NeuronModel_getExtensionOperationType(NeuronModel* model, const char* extensionName,
                                                 uint16_t operationCodeWithinExtension,
                                                 int32_t* type) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_getExtensionOperationType);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, extensionName, operationCodeWithinExtension, type);
}

inline int NeuronModel_setOperandExtensionData(NeuronModel* model, int32_t index, const void* data,
                                               size_t length) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_setOperandExtensionData);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, index, data, length);
}

inline int NeuronCompilation_createForBatch(NeuronModel* model, NeuronCompilation** compilation) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_createForBatch);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, compilation);
}

inline int NeuronModel_restoreFromCompiledNetworkV2(NeuronModel** model,
                                                    NeuronCompilation** compilation,
                                                    const void* buffer, const size_t size,
                                                    const CompilationType& type) {
    LOAD_ADAPTER_FUNCTION(NeuronModel_restoreFromCompiledNetworkV2);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, compilation, buffer, size, type);
}

inline int NeuronExecution_setRunnerPoolSize(NeuronExecution* execution, uint8_t numRunners) {
    LOAD_ADAPTER_FUNCTION(NeuronExecution_setRunnerPoolSize);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(execution, numRunners);
}

inline int NeuronExecution_setBatchDone(NeuronExecution* execution) {
    LOAD_ADAPTER_FUNCTION(NeuronExecution_setBatchDone);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(execution);
}

inline int NeuronCompilation_createWithOptions(NeuronModel* model, NeuronCompilation** compilation,
                                               const char* options) {
    LOAD_ADAPTER_FUNCTION(NeuronCompilation_createWithOptions);
    EXECUTE_ADAPTER_FUNCTION_RETURN_INT(model, compilation, options);
}
