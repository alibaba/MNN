//
//  NNAPISymbol.cpp
//  MNN
//
//  Created by MNN on 2022/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NNAPISymbol.hpp"
#include "NNAPIDefine.hpp"
#include <MNN/MNNDefine.h>
#include <dlfcn.h>

namespace MNN {

#define LOAD_SYM(NAME, API_LEVEL)                                                                   \
    NAME ## _ ## API_LEVEL = reinterpret_cast<decltype(NAME ## _ ## API_LEVEL)>(dlsym(lib, #NAME)); \
    if (NAME ## _ ## API_LEVEL == nullptr) {                                                        \
        MNN_PRINT("[NNAPI] Load symbol %s failed.", #NAME);                                         \
        return false;                                                                               \
    }

bool loadNNAPISymbol() {
    if (ANDROID_API_LEVEL < 29) {
        return false;
    }
    void *lib = dlopen("libneuralnetworks.so", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        return false;
    }
    LOAD_SYM(ANeuralNetworksModel_getSupportedOperationsForDevices, 29);
    LOAD_SYM(ANeuralNetworks_getDeviceCount, 29);
    LOAD_SYM(ANeuralNetworks_getDevice, 29);
    LOAD_SYM(ANeuralNetworksDevice_getName, 29);
    LOAD_SYM(ANeuralNetworksDevice_getType, 29);
    LOAD_SYM(ANeuralNetworksCompilation_createForDevices, 29);
    LOAD_SYM(ANeuralNetworksExecution_compute, 29);
    LOAD_SYM(ANeuralNetworksBurst_create, 29);
    LOAD_SYM(ANeuralNetworksBurst_free, 29);
    LOAD_SYM(ANeuralNetworksExecution_burstCompute, 29);
    LOAD_SYM(ANeuralNetworksModel_create, 27);
    LOAD_SYM(ANeuralNetworksModel_free, 27);
    LOAD_SYM(ANeuralNetworksModel_finish, 27);
    LOAD_SYM(ANeuralNetworksModel_addOperand, 27);
    LOAD_SYM(ANeuralNetworksModel_setOperandValue, 27);
    LOAD_SYM(ANeuralNetworksModel_setOperandSymmPerChannelQuantParams, 29);
    LOAD_SYM(ANeuralNetworksModel_addOperation, 27);
    LOAD_SYM(ANeuralNetworksModel_identifyInputsAndOutputs, 27);
    LOAD_SYM(ANeuralNetworksCompilation_create, 27);
    LOAD_SYM(ANeuralNetworksCompilation_free, 27);
    LOAD_SYM(ANeuralNetworksCompilation_setPreference, 27);
    LOAD_SYM(ANeuralNetworksCompilation_finish, 27);
    LOAD_SYM(ANeuralNetworksExecution_create, 27);
    LOAD_SYM(ANeuralNetworksExecution_free, 27);
    LOAD_SYM(ANeuralNetworksExecution_setInput, 27);
    LOAD_SYM(ANeuralNetworksExecution_setInputFromMemory, 27);
    LOAD_SYM(ANeuralNetworksExecution_setOutput, 27);
    LOAD_SYM(ANeuralNetworksExecution_setOutputFromMemory, 27);
    LOAD_SYM(ANeuralNetworksExecution_startCompute, 27);
    LOAD_SYM(ANeuralNetworksEvent_wait, 27);
    LOAD_SYM(ANeuralNetworksEvent_free, 27);
    LOAD_SYM(ANeuralNetworksDevice_getVersion, 29);
    LOAD_SYM(ANeuralNetworksMemory_createFromAHardwareBuffer, 29);
    LOAD_SYM(ANeuralNetworksMemory_createFromFd, 27);
    LOAD_SYM(ANeuralNetworksMemory_free, 27);
    LOAD_SYM(ANeuralNetworksExecution_setMeasureTiming, 29);
    LOAD_SYM(ANeuralNetworksExecution_getDuration, 29);
    return true;
}
MNN_ANeuralNetworksModel_getSupportedOperationsForDevices *ANeuralNetworksModel_getSupportedOperationsForDevices_29 = nullptr;
MNN_ANeuralNetworks_getDeviceCount *ANeuralNetworks_getDeviceCount_29 = nullptr;
MNN_ANeuralNetworks_getDevice *ANeuralNetworks_getDevice_29 = nullptr;
MNN_ANeuralNetworksDevice_getName *ANeuralNetworksDevice_getName_29 = nullptr;
MNN_ANeuralNetworksDevice_getType *ANeuralNetworksDevice_getType_29 = nullptr;
MNN_ANeuralNetworksCompilation_createForDevices *ANeuralNetworksCompilation_createForDevices_29 = nullptr;
MNN_ANeuralNetworksExecution_compute *ANeuralNetworksExecution_compute_29 = nullptr;
MNN_ANeuralNetworksBurst_create *ANeuralNetworksBurst_create_29 = nullptr;
MNN_ANeuralNetworksBurst_free *ANeuralNetworksBurst_free_29 = nullptr;
MNN_ANeuralNetworksExecution_burstCompute *ANeuralNetworksExecution_burstCompute_29 = nullptr;
MNN_ANeuralNetworksModel_create *ANeuralNetworksModel_create_27 = nullptr;
MNN_ANeuralNetworksModel_finish *ANeuralNetworksModel_finish_27 = nullptr;
MNN_ANeuralNetworksModel_free *ANeuralNetworksModel_free_27 = nullptr;
MNN_ANeuralNetworksModel_addOperand *ANeuralNetworksModel_addOperand_27 = nullptr;
MNN_ANeuralNetworksModel_setOperandValue *ANeuralNetworksModel_setOperandValue_27 = nullptr;
MNN_ANeuralNetworksModel_setOperandSymmPerChannelQuantParams *ANeuralNetworksModel_setOperandSymmPerChannelQuantParams_29;
MNN_ANeuralNetworksModel_addOperation *ANeuralNetworksModel_addOperation_27 = nullptr;
MNN_ANeuralNetworksModel_identifyInputsAndOutputs *ANeuralNetworksModel_identifyInputsAndOutputs_27 = nullptr;
MNN_ANeuralNetworksCompilation_create *ANeuralNetworksCompilation_create_27 = nullptr;
MNN_ANeuralNetworksCompilation_free *ANeuralNetworksCompilation_free_27 = nullptr;
MNN_ANeuralNetworksCompilation_setPreference *ANeuralNetworksCompilation_setPreference_27 = nullptr;
MNN_ANeuralNetworksCompilation_finish *ANeuralNetworksCompilation_finish_27 = nullptr;
MNN_ANeuralNetworksExecution_create *ANeuralNetworksExecution_create_27 = nullptr;
MNN_ANeuralNetworksExecution_free *ANeuralNetworksExecution_free_27 = nullptr;
MNN_ANeuralNetworksExecution_setInput *ANeuralNetworksExecution_setInput_27 = nullptr;
MNN_ANeuralNetworksExecution_setInputFromMemory *ANeuralNetworksExecution_setInputFromMemory_27 = nullptr;
MNN_ANeuralNetworksExecution_setOutput *ANeuralNetworksExecution_setOutput_27 = nullptr;
MNN_ANeuralNetworksExecution_setOutputFromMemory *ANeuralNetworksExecution_setOutputFromMemory_27 = nullptr;
MNN_ANeuralNetworksExecution_startCompute *ANeuralNetworksExecution_startCompute_27 = nullptr;
MNN_ANeuralNetworksEvent_wait *ANeuralNetworksEvent_wait_27 = nullptr;
MNN_ANeuralNetworksEvent_free *ANeuralNetworksEvent_free_27 = nullptr;
MNN_ANeuralNetworksDevice_getVersion *ANeuralNetworksDevice_getVersion_29 = nullptr;
MNN_ANeuralNetworksMemory_createFromAHardwareBuffer *ANeuralNetworksMemory_createFromAHardwareBuffer_29 = nullptr;
MNN_ANeuralNetworksMemory_createFromFd *ANeuralNetworksMemory_createFromFd_27 = nullptr;
MNN_ANeuralNetworksMemory_free *ANeuralNetworksMemory_free_27 = nullptr;
MNN_ANeuralNetworksExecution_setMeasureTiming *ANeuralNetworksExecution_setMeasureTiming_29 = nullptr;
MNN_ANeuralNetworksExecution_getDuration *ANeuralNetworksExecution_getDuration_29 = nullptr;
}