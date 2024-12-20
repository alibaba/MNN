//
//  ModuleBasic.cpp
//  MNN
//
//  Created by MNN on 2021/10/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "rapidjson/document.h"
#include "core/MemoryFormater.h"
#include <fstream>
#include <sstream>
#include <numeric>
#include "ExprDebug.hpp"
#include "MNN/MNNSharedContext.h"
using namespace MNN::Express;
using namespace MNN;

#ifdef __ANDROID__
#include <dlfcn.h>
#include <android/hardware_buffer.h>

/*
Ref from 
https://android.googlesource.com/platform/external/libchrome/+/refs/tags/aml_res_331314010/base/android/android_hardware_buffer_compat.h
*/
using PFAHardwareBuffer_allocate = int (*)(const AHardwareBuffer_Desc* desc,
                                            AHardwareBuffer** outBuffer);
using PFAHardwareBuffer_acquire = void (*)(AHardwareBuffer* buffer);
using PFAHardwareBuffer_describe = void (*)(const AHardwareBuffer* buffer,
                                            AHardwareBuffer_Desc* outDesc);
using PFAHardwareBuffer_lock = int (*)(AHardwareBuffer* buffer,
                                       uint64_t usage,
                                       int32_t fence,
                                       const ARect* rect,
                                       void** outVirtualAddress);
using PFAHardwareBuffer_recvHandleFromUnixSocket =
    int (*)(int socketFd, AHardwareBuffer** outBuffer);
using PFAHardwareBuffer_release = void (*)(AHardwareBuffer* buffer);
using PFAHardwareBuffer_sendHandleToUnixSocket =
    int (*)(const AHardwareBuffer* buffer, int socketFd);
using PFAHardwareBuffer_unlock = int (*)(AHardwareBuffer* buffer,
                                         int32_t* fence);

class AndroidHardwareBufferCompat {
 public:
  bool IsSupportAvailable() const {
    return true;
  }
  AndroidHardwareBufferCompat();
  int Allocate(const AHardwareBuffer_Desc* desc, AHardwareBuffer** outBuffer);
  void Acquire(AHardwareBuffer* buffer);
  void Describe(const AHardwareBuffer* buffer, AHardwareBuffer_Desc* outDesc);
  int Lock(AHardwareBuffer* buffer,
           uint64_t usage,
           int32_t fence,
           const ARect* rect,
           void** out_virtual_address);
  int RecvHandleFromUnixSocket(int socketFd, AHardwareBuffer** outBuffer);
  void Release(AHardwareBuffer* buffer);
  int SendHandleToUnixSocket(const AHardwareBuffer* buffer, int socketFd);
  int Unlock(AHardwareBuffer* buffer, int32_t* fence);
 private:
  PFAHardwareBuffer_allocate allocate_;
  PFAHardwareBuffer_acquire acquire_;
  PFAHardwareBuffer_describe describe_;
  PFAHardwareBuffer_lock lock_;
  PFAHardwareBuffer_recvHandleFromUnixSocket recv_handle_;
  PFAHardwareBuffer_release release_;
  PFAHardwareBuffer_sendHandleToUnixSocket send_handle_;
  PFAHardwareBuffer_unlock unlock_;
};
#define DCHECK(x) MNN_ASSERT(x)
AndroidHardwareBufferCompat::AndroidHardwareBufferCompat() {
  // TODO(klausw): If the Chromium build requires __ANDROID_API__ >= 26 at some
  // point in the future, we could directly use the global functions instead of
  // dynamic loading. However, since this would be incompatible with pre-Oreo
  // devices, this is unlikely to happen in the foreseeable future, so just
  // unconditionally use dynamic loading.
  // cf. base/android/linker/modern_linker_jni.cc
  void* main_dl_handle = dlopen(nullptr, RTLD_NOW);
  *reinterpret_cast<void**>(&allocate_) =
      dlsym(main_dl_handle, "AHardwareBuffer_allocate");
  DCHECK(allocate_);
  *reinterpret_cast<void**>(&acquire_) =
      dlsym(main_dl_handle, "AHardwareBuffer_acquire");
  DCHECK(acquire_);
  *reinterpret_cast<void**>(&describe_) =
      dlsym(main_dl_handle, "AHardwareBuffer_describe");
  DCHECK(describe_);
  *reinterpret_cast<void**>(&lock_) =
      dlsym(main_dl_handle, "AHardwareBuffer_lock");
  DCHECK(lock_);
  *reinterpret_cast<void**>(&recv_handle_) =
      dlsym(main_dl_handle, "AHardwareBuffer_recvHandleFromUnixSocket");
  DCHECK(recv_handle_);
  *reinterpret_cast<void**>(&release_) =
      dlsym(main_dl_handle, "AHardwareBuffer_release");
  DCHECK(release_);
  *reinterpret_cast<void**>(&send_handle_) =
      dlsym(main_dl_handle, "AHardwareBuffer_sendHandleToUnixSocket");
  DCHECK(send_handle_);
  *reinterpret_cast<void**>(&unlock_) =
      dlsym(main_dl_handle, "AHardwareBuffer_unlock");
  DCHECK(unlock_);
}

int AndroidHardwareBufferCompat::Allocate(const AHardwareBuffer_Desc* desc,
                                           AHardwareBuffer** out_buffer) {
  DCHECK(IsSupportAvailable());
  return allocate_(desc, out_buffer);
}
void AndroidHardwareBufferCompat::Acquire(AHardwareBuffer* buffer) {
  DCHECK(IsSupportAvailable());
  acquire_(buffer);
}
void AndroidHardwareBufferCompat::Describe(const AHardwareBuffer* buffer,
                                           AHardwareBuffer_Desc* out_desc) {
  DCHECK(IsSupportAvailable());
  describe_(buffer, out_desc);
}
int AndroidHardwareBufferCompat::Lock(AHardwareBuffer* buffer,
                                      uint64_t usage,
                                      int32_t fence,
                                      const ARect* rect,
                                      void** out_virtual_address) {
  DCHECK(IsSupportAvailable());
  return lock_(buffer, usage, fence, rect, out_virtual_address);
}
int AndroidHardwareBufferCompat::RecvHandleFromUnixSocket(
    int socket_fd,
    AHardwareBuffer** out_buffer) {
  DCHECK(IsSupportAvailable());
  return recv_handle_(socket_fd, out_buffer);
}
void AndroidHardwareBufferCompat::Release(AHardwareBuffer* buffer) {
  DCHECK(IsSupportAvailable());
  release_(buffer);
}
int AndroidHardwareBufferCompat::SendHandleToUnixSocket(
    const AHardwareBuffer* buffer,
    int socket_fd) {
  DCHECK(IsSupportAvailable());
  return send_handle_(buffer, socket_fd);
}
int AndroidHardwareBufferCompat::Unlock(AHardwareBuffer* buffer,
                                        int32_t* fence) {
  DCHECK(IsSupportAvailable());
  return unlock_(buffer, fence);
}

static std::shared_ptr<AndroidHardwareBufferCompat> gFunction;

static AHardwareBuffer* creatAHardwareBuffer(int width, int height, void *data){
    // 创建和初始化硬件缓冲区
    AHardwareBuffer_Desc bufferDesc = {};
    bufferDesc.width = width;
    bufferDesc.height = height;
    bufferDesc.layers = 1;
    bufferDesc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
    bufferDesc.usage = AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE;

    AHardwareBuffer* buffer = nullptr;
    int result = gFunction->Allocate(&bufferDesc, &buffer);
    if(result != 0) {
        // Handle allocation error
        MNN_PRINT("alloc AHardwareBuffer failed   %d\n", result);
    }
        
    if(nullptr != data){
        void* map = nullptr;
        ARect rect = { 0, 0, width, height };  // Define the region to lock
        result = gFunction->Lock(buffer, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, &rect, &map);
        if (result != 0) {
            // Handle lock failure
            MNN_PRINT("Handle lock failed\n");
        }
        if (map) {
            // Now write your pixel data to 'data'
            // For example, fill it with a solid color:
            memcpy(map, data, width * height * 4); // Assuming RGBA8888 format
        }
            
        gFunction->Unlock(buffer, nullptr);
    }
    return buffer;
}
static void copyDataFromAHardWareBuffer(AHardwareBuffer* buffer, int width, int height, void *data){
    int result = 0;
    if(nullptr != data){
        void* map = nullptr;
        ARect rect = { 0, 0, width, height };  // Define the region to lock
        result = gFunction->Lock(buffer, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1, &rect, &map);
        if (result != 0) {
            MNN_PRINT("Handle lock failed\n");
        }
        if (map) {
            memcpy(data, map, width * height * 4);
        }
            
        gFunction->Unlock(buffer, nullptr);
    }
}
static void ReleaseAHardWareBuffer(AHardwareBuffer* buffer){
    if(buffer != nullptr){
        gFunction->Release(buffer);
    }
}
#endif

int main(int argc, char *argv[]) {
    if (argc < 3) {
        MNN_ERROR("Usage: ./GpuInterTest.out ${test.mnn} ${Dir} [testMode] [forwardType] [numberThread] [precision | memory]\n");
        return 0;
    }
    std::string modelName = argv[1];
    std::string directName = argv[2];
    MNN_PRINT("Test %s from input info: %s\n", modelName.c_str(), directName.c_str());
    std::map<std::string, float> inputInfo;
    std::map<std::string, std::vector<int>> inputShape;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    int repeatNumber = 1;
    bool shapeMutable = true;
    std::vector<VARP> inputs;
    std::vector<VARP> outputs;
    if (inputNames.empty()) {
        rapidjson::Document document;
        std::ostringstream jsonNameOs;
        jsonNameOs << directName << "/input.json";
        std::ifstream fileNames(jsonNameOs.str().c_str());
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
        if (document.HasMember("inputs")) {
            auto inputsInfo = document["inputs"].GetArray();
            for (auto iter = inputsInfo.begin(); iter !=inputsInfo.end(); iter++) {
                auto obj = iter->GetObject();
                std::string name = obj["name"].GetString();
                inputNames.emplace_back(name);
                MNN_PRINT("%s\n", name.c_str());
                if (obj.HasMember("value")) {
                    float value = obj["value"].GetFloat();
                    inputInfo.insert(std::make_pair(name, value));
                }
                if (obj.HasMember("shape")) {
                    auto dims = obj["shape"].GetArray();
                    std::vector<int> shapes;
                    for (auto iter = dims.begin(); iter != dims.end(); iter++) {
                        shapes.emplace_back(iter->GetInt());
                    }
                    inputShape.insert(std::make_pair(name, shapes));
                }
            }
        }
        if (document.HasMember("outputs")) {
            auto array = document["outputs"].GetArray();
            for (auto iter = array.begin(); iter !=array.end(); iter++) {
                std::string name = iter->GetString();
                MNN_PRINT("output: %s\n", name.c_str());
                outputNames.emplace_back(name);
            }
        }
        if (document.HasMember("shapeMutable")) {
            shapeMutable = document["shapeMutable"].GetBool();
        }
        if (document.HasMember("repeat")) {
            repeatNumber = document["repeat"].GetInt();
        }
    }
    int testMode = 0;
    //testMode = 0 AhardwareBuffer
    if(argc > 3){
        testMode = atoi(argv[3]);
        MNN_PRINT("Use extra forward type: %d(0:AhardwareBuffer)\n", testMode);
    }

    auto type = MNN_FORWARD_CPU;
    if (argc > 4) {
        type = (MNNForwardType)atoi(argv[4]);
        MNN_PRINT("Use extra forward type: %d\n", type);
    }

    // Default single thread
    int modeNum = 1;
    if (argc > 5) {
        modeNum = ::atoi(argv[5]);
    }

    int precision = BackendConfig::Precision_Normal;
    int memory = BackendConfig::Memory_Normal;
    if (argc > 6) {
        int mask = atoi(argv[6]);
        precision = mask % 4;
        memory = (mask / 4) % 4;
    }
    const char* cacheFileName = ".tempcache";
    FUNC_PRINT(precision);
    FUNC_PRINT(memory);
    FUNC_PRINT_ALL(cacheFileName, s);
    // create session
    MNN::ScheduleConfig config;
    config.type      = type;
    /*modeNum means gpuMode for GPU usage, Or means numThread for CPU usage.*/
    config.numThread = modeNum;
    // If type not fount, let it failed
    config.backupType = type;
    BackendConfig backendConfig;
    backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(precision);
    backendConfig.memory = static_cast<MNN::BackendConfig::MemoryMode>(memory);
    config.backendConfig     = &backendConfig;

    MNN::Express::Module::Config mConfig;
    mConfig.shapeMutable = shapeMutable;
#ifdef __ANDROID__
    gFunction.reset(new AndroidHardwareBufferCompat);
    std::vector<AHardwareBuffer*> AHardwarePtrInputVec;
    std::vector<AHardwareBuffer*> AHardwarePtrOutputVec;
#endif

    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setCache(cacheFileName);
    std::shared_ptr<Module> net;
    {
        AUTOTIME;
        net.reset(Module::load(inputNames, outputNames, modelName.c_str(), rtmgr, &mConfig));
        if (net == nullptr) {
            MNN_PRINT("Error: can't load module\n");
            return 0;
        }
    }
    auto mInfo = net->getInfo();
#ifdef __ANDROID__
    AHardwarePtrInputVec.resize(mInfo->inputs.size());
    AHardwarePtrOutputVec.resize(outputNames.size());
#endif
    if (inputs.empty()) {
        inputs.resize(mInfo->inputs.size());
        for (int i=0; i<inputs.size(); ++i) {
            inputs[i] = _Input(mInfo->inputs[i].dim, mInfo->inputs[i].order, mInfo->inputs[i].type);
        }
        for (int i=0; i<inputs.size(); ++i) {
            auto inputName = inputNames[i];
            // Resize
            auto info = inputs[i]->getInfo();
            int width = info->dim[3], height = info->dim[2], channel = info->dim[1];
            auto shapeIter = inputShape.find(inputName);
            if (shapeIter != inputShape.end()) {
                auto s = shapeIter->second;
                inputs[i] = _Input(s, mInfo->inputs[i].order, mInfo->inputs[i].type);
                width = s[3];
                height = s[2];
                channel = s[1];
            }
            // set input device ptr
#ifdef __ANDROID__
            // OpenGL Texture defaultFormat NC4HW4
            if(testMode == 0){
                width = width * ((channel + 3) / 4);
                AHardwarePtrInputVec[i] = creatAHardwareBuffer(width,height,nullptr);
                volatile uint64_t value = (uint64_t)AHardwarePtrInputVec[i];
                inputs[i]->setDevicePtr((void*)value, MNN_MEMORY_AHARDWAREBUFFER);
            }
#endif
        }
    }

    bool modelError = false;
    for (int repeat = 0; repeat < repeatNumber; ++repeat) {
        AUTOTIME;
        auto outputs = net->onForward(inputs);
        if (outputs.empty()) {
            MNN_ERROR("Error in forward\n");
            return 0;
        }
        for (int i=0; i<outputNames.size(); ++i) {
            auto info = outputs[i]->getInfo();
            int width = info->dim[3], height = info->dim[2], channel = info->dim[1];
            // copy output to device ptr
#ifdef __ANDROID__
            if(testMode == 0){
                AHardwarePtrOutputVec[i] = creatAHardwareBuffer(width,height,nullptr);
                volatile uint64_t value = (uint64_t)AHardwarePtrOutputVec[i];
                outputs[i]->copyToDevicePtr((void*)value, MNN_MEMORY_AHARDWAREBUFFER);
            }
#endif
        }

        // Print module's memory
        float memoryInMB = 0.0f;
        rtmgr->getInfo(Interpreter::MEMORY, &memoryInMB);
        FUNC_PRINT_ALL(memoryInMB, f);
    }
#ifdef __ANDROID__
    if(testMode == 1){
        for(int i = 0; i < AHardwarePtrInputVec.size(); ++i){
            ReleaseAHardWareBuffer(AHardwarePtrInputVec[i]);
        }
        for(int i = 0; i < AHardwarePtrOutputVec.size(); ++i){
            ReleaseAHardWareBuffer(AHardwarePtrOutputVec[i]);
        }
    }
#endif
    rtmgr->updateCache();
    return 0;
}

