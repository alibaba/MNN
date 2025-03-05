//
//  NeuronAdapterSymbol.cpp
//  MNN
//
//  Created by MNN on 2022/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NeuronAdapterSymbol.hpp"
#include "NeuronAdapterDefine.hpp"
#include <MNN/MNNDefine.h>
#include <dlfcn.h>

namespace MNN {

#define LOAD_SYM(NAME, API_LEVEL)                                                                   \
    NAME ## _ ## API_LEVEL = reinterpret_cast<decltype(NAME ## _ ## API_LEVEL)>(dlsym(lib, #NAME)); \
    if (NAME ## _ ## API_LEVEL == nullptr) {                                                        \
        MNN_PRINT("[NeuronAdapter] Load symbol %s failed.", #NAME);                                         \
        return false;                                                                               \
    }

bool loadNeuronAdapterSymbol() {
    if (ANDROID_API_LEVEL < 29) {
        return false;
    }
    void *lib = dlopen("libneuronusdk_adapter.mtk.so", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        return false;
    }
    LOAD_SYM(NeuronModel_getSupportedOperationsForDevices, 29);
    LOAD_SYM(Neuron_getDeviceCount, 29);
    LOAD_SYM(Neuron_getDevice, 29);
    LOAD_SYM(NeuronDevice_getName, 29);
    LOAD_SYM(NeuronCompilation_createForDevices, 29);
    LOAD_SYM(NeuronExecution_compute, 29);
    LOAD_SYM(NeuronModel_create, 27);
    LOAD_SYM(NeuronModel_free, 27);
    LOAD_SYM(NeuronModel_finish, 27);
    LOAD_SYM(NeuronModel_addOperand, 27);
    LOAD_SYM(NeuronModel_setOperandValue, 27);
    LOAD_SYM(NeuronModel_setOperandSymmPerChannelQuantParams, 29);
    LOAD_SYM(NeuronModel_addOperation, 27);
    LOAD_SYM(NeuronModel_identifyInputsAndOutputs, 27);
    LOAD_SYM(NeuronCompilation_create, 27);
    LOAD_SYM(NeuronCompilation_free, 27);
    LOAD_SYM(NeuronCompilation_setPreference, 27);
    LOAD_SYM(NeuronCompilation_finish, 27);
    LOAD_SYM(NeuronExecution_create, 27);
    LOAD_SYM(NeuronExecution_free, 27);
    LOAD_SYM(NeuronExecution_setInput, 27);
    LOAD_SYM(NeuronExecution_setInputFromMemory, 27);
    LOAD_SYM(NeuronExecution_setOutput, 27);
    LOAD_SYM(NeuronExecution_setOutputFromMemory, 27);
    LOAD_SYM(NeuronEvent_wait, 27);
    LOAD_SYM(NeuronEvent_free, 27);
    LOAD_SYM(NeuronMemory_createFromAHardwareBuffer, 29);
    LOAD_SYM(NeuronMemory_createFromFd, 27);
    LOAD_SYM(NeuronMemory_free, 27);
    return true;
}
MNN_NeuronModel_getSupportedOperationsForDevices *NeuronModel_getSupportedOperationsForDevices_29 = nullptr;
MNN_Neuron_getDeviceCount *Neuron_getDeviceCount_29 = nullptr;
MNN_Neuron_getDevice *Neuron_getDevice_29 = nullptr;
MNN_NeuronDevice_getName *NeuronDevice_getName_29 = nullptr;
MNN_NeuronCompilation_createForDevices *NeuronCompilation_createForDevices_29 = nullptr;
MNN_NeuronExecution_compute *NeuronExecution_compute_29 = nullptr;
MNN_NeuronModel_create *NeuronModel_create_27 = nullptr;
MNN_NeuronModel_finish *NeuronModel_finish_27 = nullptr;
MNN_NeuronModel_free *NeuronModel_free_27 = nullptr;
MNN_NeuronModel_addOperand *NeuronModel_addOperand_27 = nullptr;
MNN_NeuronModel_setOperandValue *NeuronModel_setOperandValue_27 = nullptr;
MNN_NeuronModel_setOperandSymmPerChannelQuantParams *NeuronModel_setOperandSymmPerChannelQuantParams_29;
MNN_NeuronModel_addOperation *NeuronModel_addOperation_27 = nullptr;
MNN_NeuronModel_identifyInputsAndOutputs *NeuronModel_identifyInputsAndOutputs_27 = nullptr;
MNN_NeuronCompilation_create *NeuronCompilation_create_27 = nullptr;
MNN_NeuronCompilation_free *NeuronCompilation_free_27 = nullptr;
MNN_NeuronCompilation_setPreference *NeuronCompilation_setPreference_27 = nullptr;
MNN_NeuronCompilation_finish *NeuronCompilation_finish_27 = nullptr;
MNN_NeuronExecution_create *NeuronExecution_create_27 = nullptr;
MNN_NeuronExecution_free *NeuronExecution_free_27 = nullptr;
MNN_NeuronExecution_setInput *NeuronExecution_setInput_27 = nullptr;
MNN_NeuronExecution_setInputFromMemory *NeuronExecution_setInputFromMemory_27 = nullptr;
MNN_NeuronExecution_setOutput *NeuronExecution_setOutput_27 = nullptr;
MNN_NeuronExecution_setOutputFromMemory *NeuronExecution_setOutputFromMemory_27 = nullptr;
MNN_NeuronExecution_compute *NeuronExecution_compute_27 = nullptr;
MNN_NeuronEvent_wait *NeuronEvent_wait_27 = nullptr;
MNN_NeuronEvent_free *NeuronEvent_free_27 = nullptr;
MNN_NeuronMemory_createFromAHardwareBuffer *NeuronMemory_createFromAHardwareBuffer_29 = nullptr;
MNN_NeuronMemory_createFromFd *NeuronMemory_createFromFd_27 = nullptr;
MNN_NeuronMemory_free *NeuronMemory_free_27 = nullptr;
}