//
//  ReplaceTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __ANDROID__
#include <dlfcn.h>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include <MNN/expr/Module.hpp>
#include "TestUtils.h"
#include <android/hardware_buffer.h>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

using namespace MNN;
using namespace MNN::Express;

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
    return mIsSupportAvailable;
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
  bool mIsSupportAvailable = true;
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
  if(nullptr == allocate_){
      mIsSupportAvailable = false;
  }
  *reinterpret_cast<void**>(&acquire_) =
      dlsym(main_dl_handle, "AHardwareBuffer_acquire");
  if(nullptr == acquire_){
      mIsSupportAvailable = false;
  }
  *reinterpret_cast<void**>(&describe_) =
      dlsym(main_dl_handle, "AHardwareBuffer_describe");
  if(nullptr == describe_){
      mIsSupportAvailable = false;
  }
  *reinterpret_cast<void**>(&lock_) =
      dlsym(main_dl_handle, "AHardwareBuffer_lock");
  if(nullptr == lock_){
      mIsSupportAvailable = false;
  }
  *reinterpret_cast<void**>(&recv_handle_) =
      dlsym(main_dl_handle, "AHardwareBuffer_recvHandleFromUnixSocket");
  if(nullptr == recv_handle_){
      mIsSupportAvailable = false;
  }
  *reinterpret_cast<void**>(&release_) =
      dlsym(main_dl_handle, "AHardwareBuffer_release");
  if(nullptr == release_){
      mIsSupportAvailable = false;
  }
  *reinterpret_cast<void**>(&send_handle_) =
      dlsym(main_dl_handle, "AHardwareBuffer_sendHandleToUnixSocket");
  if(nullptr == send_handle_){
      mIsSupportAvailable = false;
  }
  *reinterpret_cast<void**>(&unlock_) =
      dlsym(main_dl_handle, "AHardwareBuffer_unlock");
  if(nullptr == unlock_){
      mIsSupportAvailable = false;
  }
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

static AHardwareBuffer* creatAHardwareBufferRGBA(int width, int height, void *data){
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
        MNN_ERROR("alloc AHardwareBuffer failed   %d\n", result);
    }
    
    if(nullptr != data){
        void* map = nullptr;
        ARect rect = { 0, 0, width, height };  // Define the region to lock
        result = gFunction->Lock(buffer, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, &rect, &map);
        if (result != 0) {
            MNN_ERROR("Handle lock failed\n");
        }
        AHardwareBuffer_Desc Desc = {};
        gFunction->Describe(buffer, &Desc);
        int stride = Desc.stride;
        if (map) {
            for(int i = 0; i < height; ++i){
                unsigned char* src = (unsigned char*)data + i * width * 4;
                unsigned char* dst = (unsigned char*)map + i * stride * 4;
                memcpy(dst, src, width * 4);
            }
        }
        
        gFunction->Unlock(buffer, nullptr);
    }
    return buffer;
}

static AHardwareBuffer* creatAHardwareBufferYUV420(int width, int height, void *data){
    // 创建和初始化硬件缓冲区
    AHardwareBuffer_Desc bufferDesc = {};
    bufferDesc.width = width;
    bufferDesc.height = height;
    bufferDesc.layers = 1;
    bufferDesc.format = AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420;
    bufferDesc.usage = AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE;

    AHardwareBuffer* buffer = nullptr;
    int result = gFunction->Allocate(&bufferDesc, &buffer);
    if(result != 0) {
        MNN_ERROR("alloc AHardwareBuffer failed   %d\n", result);
    }
    
    if(nullptr != data){
        void* map = nullptr;
        ARect rect = { 0, 0, width, height };  // Define the region to lock
        result = gFunction->Lock(buffer, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, &rect, &map);
        if (result != 0) {
            MNN_ERROR("Handle lock failed\n");
        }
        AHardwareBuffer_Desc Desc = {};
        gFunction->Describe(buffer, &Desc);
        int stride = Desc.stride;
        if (map) {
            for(int i = 0; i < height; ++i){
                unsigned char* src = (unsigned char*)data + i * width;
                unsigned char* dst = (unsigned char*)map + i * stride;
                memcpy(dst, src, width);
            }
        }
        
        gFunction->Unlock(buffer, nullptr);
    }
    return buffer;
}

static void ReleaseAHardWareBuffer(AHardwareBuffer* buffer){
    gFunction->Release(buffer);
}

static void copyDataFromAHardWareBufferRGBA(AHardwareBuffer* buffer, int width, int height, void *data){
    int result = 0;
    if(nullptr != data){
        void* map = nullptr;
        ARect rect = { 0, 0, width, height };  // Define the region to lock
        result = gFunction->Lock(buffer, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1, &rect, &map);
        AHardwareBuffer_Desc bufferDesc = {};
        gFunction->Describe(buffer, &bufferDesc);
        int stride = bufferDesc.stride;
        if (result != 0) {
            MNN_ERROR("Handle lock failed\n");
        }
        if (map) {
            for(int i = 0; i < height; ++i){
                unsigned char* dst = (unsigned char*)data + i * width * 4;
                unsigned char* src = (unsigned char*)map + i * stride * 4;
                memcpy(dst, src, width * 4);
            }
        }
        
        gFunction->Unlock(buffer, nullptr);
    }
}

static void copyDataFromAHardWareBufferYUV420(AHardwareBuffer* buffer, int width, int height, void *data){
    int result = 0;
    if(nullptr != data){
        void* map = nullptr;
        ARect rect = { 0, 0, width, height };  // Define the region to lock
        result = gFunction->Lock(buffer, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1, &rect, &map);
        AHardwareBuffer_Desc bufferDesc = {};
        gFunction->Describe(buffer, &bufferDesc);
        int stride = bufferDesc.stride;
        if (result != 0) {
            MNN_ERROR("Handle lock failed\n");
        }
        if (map) {
            for(int i = 0; i < height; ++i){
                unsigned char* dst = (unsigned char*)data + i * width;
                unsigned char* src = (unsigned char*)map + i * stride;
                memcpy(dst, src, width);
            }
        }
        
        gFunction->Unlock(buffer, nullptr);
    }
}

static bool checkvalue(const float* ref, const unsigned char* out, int size){
    for(int i = 0; i < size; ++i){
        if(ref[i] != (float)out[i]){
            MNN_ERROR("%d:  ref %f != out %f\n", i, ref[i], (float)out[i]);
            return false;
        }
    }
    return true;
}

const int width = 1280;
const int height = 720;
static std::shared_ptr<Module> _createModel(int channel) {
    auto x = _Input({1, channel, height, width}, NCHW, halide_type_of<float>());
    x->setName("Input");
    auto y = _Transpose(x, {0, 1, 3, 2});
    y->setName("Transpose");
    std::unique_ptr<NetT> net(new NetT);
    Variable::save({y}, net.get());
    flatbuffers::FlatBufferBuilder builder;
    auto len = MNN::Net::Pack(builder, net.get());
    builder.Finish(len);
    return std::shared_ptr<Module>(Module::load({"Input"}, {"Transpose"}, builder.GetBufferPointer(), builder.GetSize()));
}
// Test prepareCompute for dynamic-graph usage
class AhardWareBufferTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        if (nullptr == gFunction) {
            gFunction.reset(new AndroidHardwareBufferCompat);
        }
        if (MNN_FORWARD_OPENCL != getCurrentType()) {
            MNN_ERROR("Currently forwardtype[%d] run sharedmem/AhardWareBuffer has error, skip it\n", getCurrentType());
            return true;
        }
        // test rgbainput
        {
            int channel = 3;
            auto net = _createModel(channel);
            auto x = _Input({1, channel, height, width}, NCHW, halide_type_of<float>());
            unsigned char inputData[4 * height * width];
            unsigned char outputData[4 * height * width];
            for(int i = 0; i < 4 * height * width; ++i){
                inputData[i] = rand() % 255;
            }
            // ahardwarebuffer format is rgba
            {
                auto xPtr = x->writeMap<float>();
                for (int i = 0; i < channel; ++i){
                    for (int j = 0; j < height * width; ++j) {
                        xPtr[i * height * width + j] = (float)inputData[j * 4 + i];
                    }
                }
                x->unMap();
            }
            
            auto outputs = net->onForward({x});
            outputs[0] = _Convert(outputs[0], NC4HW4);
            auto refPtr = outputs[0]->readMap<float>();
            auto size = outputs[0]->getInfo()->size;
            
            auto xShared = _Input({1, channel, height, width}, NCHW, halide_type_of<float>());
            auto inputAhardwareBuffer = creatAHardwareBufferRGBA(width, height, inputData);
            volatile uint64_t inputValue = (uint64_t)inputAhardwareBuffer;
            xShared->setDevicePtr((void*)inputValue, MNN_MEMORY_AHARDWAREBUFFER);
            auto outputsShared = net->onForward({xShared});
            auto outputAhardwareBuffer = creatAHardwareBufferRGBA(height, width, nullptr);
            volatile uint64_t outputValue = (uint64_t)outputAhardwareBuffer;
            {
                outputsShared[0]->copyToDevicePtr((void*)outputValue, MNN_MEMORY_AHARDWAREBUFFER);
                copyDataFromAHardWareBufferRGBA(outputAhardwareBuffer, height, width, outputData);
                if(checkvalue(refPtr, outputData, size) == false){
                    MNN_ERROR("sharedmem/AhardWareBuffer RGBA format test failed!\n");
                    return false;
                }
            }
            
            // speed
            const auto time = 100;
            {
                MNN::Timer _t;
                for (int t = 0; t < time; ++t) {
                    x->writeMap<float>();
                    auto outputs = net->onForward({x});
                    outputs[0]->readMap<float>();
                }
                float timeCost = _t.durationInUs() / 1000.0f / (float)time;
                MNN_PRINT("cpu copy [%d, %d, %d], Avg time: %f ms\n", channel, height, width, timeCost);
            }
            {
                MNN::Timer _t;
                for (int t = 0; t < time; ++t) {
                    xShared->setDevicePtr((void*)inputValue, MNN_MEMORY_AHARDWAREBUFFER);
                    auto outputs = net->onForward({xShared});
                    outputs[0]->copyToDevicePtr((void*)outputValue, MNN_MEMORY_AHARDWAREBUFFER);
                }
                float timeCost = _t.durationInUs() / 1000.0f / (float)time;
                MNN_PRINT("shared memory copy [%d, %d, %d], Avg time: %f ms\n", channel, height, width, timeCost);
            }
            
            ReleaseAHardWareBuffer(inputAhardwareBuffer);
            ReleaseAHardWareBuffer(outputAhardwareBuffer);
        }
        
        // test yuvinput
        {
            int channel = 1;
            auto net = _createModel(channel);
            auto x = _Input({1, channel, height, width}, NCHW, halide_type_of<float>());
            //yuv420
            unsigned char inputData[height * width + height * width / 2];
            unsigned char outputData[height * width + height * width / 2];
            ::memcpy(outputData, inputData, height * width + height * width / 2);
            for(int i = 0; i < height * width + height * width / 2; ++i){
                inputData[i] = rand() % 255;
            }
            // ahardwarebuffer format is yuv420
            {
                auto xPtr = x->writeMap<float>();
                for (int i = 0; i < height * width; ++i) {
                    xPtr[i] = (float)inputData[i];
                }
                x->unMap();
            }
            
            auto outputs = net->onForward({x});
            auto refPtr = outputs[0]->readMap<float>();
            auto size = outputs[0]->getInfo()->size;
            
            auto xShared = _Input({1, channel, height, width}, NCHW, halide_type_of<float>());
            auto inputAhardwareBuffer = creatAHardwareBufferYUV420(width, height, inputData);
            volatile uint64_t inputValue = (uint64_t)inputAhardwareBuffer;
            xShared->setDevicePtr((void*)inputValue, MNN_MEMORY_AHARDWAREBUFFER);
            auto outputsShared = net->onForward({xShared});
            auto outputAhardwareBuffer = creatAHardwareBufferYUV420(height, width, nullptr);
            volatile uint64_t outputValue = (uint64_t)outputAhardwareBuffer;
            {
                outputsShared[0]->copyToDevicePtr((void*)outputValue, MNN_MEMORY_AHARDWAREBUFFER);
                copyDataFromAHardWareBufferYUV420(outputAhardwareBuffer, height, width, outputData);
                if(checkvalue(refPtr, outputData, size) == false){
                    MNN_ERROR("sharedmem/AhardWareBuffer YUV420 format test failed!\n");
                    return false;
                }
            }
            
            // speed
            const auto time = 100;
            {
                MNN::Timer _t;
                for (int t = 0; t < time; ++t) {
                    x->writeMap<float>();
                    auto outputs = net->onForward({x});
                    outputs[0]->readMap<float>();
                }
                float timeCost = _t.durationInUs() / 1000.0f / (float)time;
                MNN_PRINT("cpu copy [%d, %d, %d], Avg time: %f ms\n", channel, height, width, timeCost);
            }
            {
                MNN::Timer _t;
                for (int t = 0; t < time; ++t) {
                    xShared->setDevicePtr((void*)inputValue, MNN_MEMORY_AHARDWAREBUFFER);
                    auto outputs = net->onForward({xShared});
                    outputs[0]->copyToDevicePtr((void*)outputValue, MNN_MEMORY_AHARDWAREBUFFER);
                }
                float timeCost = _t.durationInUs() / 1000.0f / (float)time;
                MNN_PRINT("shared memory copy [%d, %d, %d], Avg time: %f ms\n", channel, height, width, timeCost);
            }
            
            ReleaseAHardWareBuffer(inputAhardwareBuffer);
            ReleaseAHardWareBuffer(outputAhardwareBuffer);
        }
        return true;
    }
};

MNNTestSuiteRegister(AhardWareBufferTest, "sharedmem/AhardWareBuffer");
#endif
