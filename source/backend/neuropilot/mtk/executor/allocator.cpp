#include <MNN/MNNDefine.h>
#include "executor/allocator.h"

#include "backend/api/neuron/NeuronAdapterShim.h"
#include "common/cpp11_compat.h"
#include "common/logging.h"

#include <android/hardware_buffer.h>
#include <fcntl.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef USE_USDK_BACKEND
static constexpr bool kUseUsdkBackend = true;
#else
static constexpr bool kUseUsdkBackend = false;
#endif

namespace mtk {
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
  return allocate_(desc, out_buffer);
}
void AndroidHardwareBufferCompat::Acquire(AHardwareBuffer* buffer) {
  acquire_(buffer);
}
void AndroidHardwareBufferCompat::Describe(const AHardwareBuffer* buffer,
                                           AHardwareBuffer_Desc* out_desc) {
  describe_(buffer, out_desc);
}
int AndroidHardwareBufferCompat::Lock(AHardwareBuffer* buffer,
                                      uint64_t usage,
                                      int32_t fence,
                                      const ARect* rect,
                                      void** out_virtual_address) {
  return lock_(buffer, usage, fence, rect, out_virtual_address);
}
int AndroidHardwareBufferCompat::RecvHandleFromUnixSocket(
    int socket_fd,
    AHardwareBuffer** out_buffer) {
  return recv_handle_(socket_fd, out_buffer);
}
void AndroidHardwareBufferCompat::Release(AHardwareBuffer* buffer) {
  release_(buffer);
}
int AndroidHardwareBufferCompat::SendHandleToUnixSocket(
    const AHardwareBuffer* buffer,
    int socket_fd) {
  return send_handle_(buffer, socket_fd);
}
int AndroidHardwareBufferCompat::Unlock(AHardwareBuffer* buffer,
                                        int32_t* fence) {
  return unlock_(buffer, fence);
}

Allocator::~Allocator() {
    // NOTE: Allocator::releaseAll() needs to be called before reaching this dtor.
}

IOBuffer Allocator::allocate(const size_t size, const bool disableTracking) {
    IOBuffer ioBuffer;
    ioBuffer.sizeBytes = size;
    ioBuffer.usedSizeBytes = size;
    if (!allocateMemory(ioBuffer)) {
        return IOBuffer(); // Empty buffer
    }
    if (disableTracking) {
        // Remove from allocation tracking so that releaseAll() won't release this buffer.
        mAllocatedBuffers.erase(ioBuffer.buffer);
    }
    return ioBuffer;
}

bool Allocator::release(void* addr) {
    if (mAllocatedBuffers.find(addr) == mAllocatedBuffers.end()) {
        LOG(ERROR) << "Unable to release the memory at " << addr << " by the allocator instance "
                   << this;
        return false;
    }
    auto& ioBuffer = mAllocatedBuffers.at(addr);
    return releaseMemory(ioBuffer);
}

void Allocator::releaseAll() {
    while (!mAllocatedBuffers.empty()) {
        auto& bufferPair = *mAllocatedBuffers.begin();
        auto& addr = bufferPair.first;
        auto& ioBuffer = bufferPair.second;
        releaseMemory(ioBuffer);
    }
}

AhwBufferAllocator::AhwBufferAllocator() {
    mFuncPtr.reset(new AndroidHardwareBufferCompat);
    if (!mFuncPtr->IsSupportAvailable()) {
        MNN_ERROR("[MNN:AhwBufferAllocator] Don't has hardware funciton ptr\n");
    }
}

AhwBufferAllocator::~AhwBufferAllocator() {
    releaseAll();
}

bool AhwBufferAllocator::allocateMemory(IOBuffer& ioBuffer) {
    const auto& size = ioBuffer.sizeBytes;
    auto usage = AHARDWAREBUFFER_USAGE_CPU_READ_RARELY | AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY;

    AHardwareBuffer_Desc ahwbDesc = {
        .width = static_cast<uint32_t>(size),
        .height = 1,
        .layers = 1,
        .format = AHARDWAREBUFFER_FORMAT_BLOB,
        .usage = usage,
        .stride = static_cast<uint32_t>(size),
    };

    AHardwareBuffer* ahwbHandle = nullptr;
    if (mFuncPtr->Allocate(&ahwbDesc, &ahwbHandle) != 0) {
        LOG(ERROR) << "Allocate AHardwareBuffer fail";
        return false;
    }

    void* bufferAddr = nullptr;
    mFuncPtr->Lock(ahwbHandle, usage, -1, NULL, &bufferAddr);

    ioBuffer.ahwbHandle = ahwbHandle;
    ioBuffer.buffer = bufferAddr;

    // USDK requires NeuronMemory
    if (kUseUsdkBackend) {
        NeuronMemory_createFromAHardwareBuffer(ioBuffer.ahwbHandle, &ioBuffer.neuronMemory);
    }

    // Add to allocated buffer record
    mAllocatedBuffers.emplace(bufferAddr, ioBuffer);
    return true;
}

bool AhwBufferAllocator::releaseMemory(IOBuffer& ioBuffer) {
    if (!ioBuffer.isAllocated()) {
        return true; // Do nothing
    }

    if (ioBuffer.neuronMemory != nullptr) {
        NeuronMemory_free(ioBuffer.neuronMemory);
    }

    mFuncPtr->Unlock(ioBuffer.ahwbHandle, nullptr);
    mFuncPtr->Release(ioBuffer.ahwbHandle);

    const auto bufferAddr = ioBuffer.buffer;

    ioBuffer.buffer = nullptr;
    ioBuffer.sizeBytes = 0;
    ioBuffer.fd = -1;

    // Remove from allocated buffer record if exist
    mAllocatedBuffers.erase(bufferAddr);

    return true;
}

} // namespace mtk
