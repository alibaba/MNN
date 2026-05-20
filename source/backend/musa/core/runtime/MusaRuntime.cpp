//
//  MusaRuntime.cpp
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "MusaRuntime.hpp"
#include "MNN/MNNSharedContext.h"
#include <MNN/ErrorCode.hpp>
#include <string.h>
#include <mutex>

namespace MNN {

MusaRuntime::MusaRuntime(int device_id) {
    mDeviceId = device_id;
    mDeviceCount = 0;
    mIsSupportedFP16 = true;
    mSupportDotInt8 = false;
    mSupportDotAccInt8 = false;
    mFlops = 4.0f;
    mIsCreateError = false;
    mThreadPerBlock = 128;
    
    // Initialize device properties with defaults
    memset(&mProp, 0, sizeof(musaDeviceProp));
    mProp.major = 7;
    mProp.minor = 0;
    mProp.multiProcessorCount = 1;
    mProp.maxThreadsPerBlock = 1024;
    mProp.sharedMemPerBlock = 49152;
    mProp.warpSize = 32;
    mProp.maxThreadsPerMultiProcessor = 2048;
    mProp.totalGlobalMem = 8 * 1024 * 1024 * 1024ULL; // 8GB default
    strcpy(mProp.name, "MUSA Stub Device");
    
    musaError_t err = musaSetDevice(mDeviceId);
    if (err != musaSuccess) {
        MNN_PRINT("MUSA device not available, using stub mode\n");
        // Don't set error - allow stub mode to continue
    }
    
    err = musaGetDeviceProperties(&mProp, mDeviceId);
    if (err == musaSuccess) {
        MNN_PRINT("MUSA Device: %s\n", mProp.name);
        MNN_PRINT("MUSA Compute Capability: %d.%d\n", mProp.major, mProp.minor);
        MNN_PRINT("MUSA Multiprocessor Count: %d\n", mProp.multiProcessorCount);
    }
    
    // Calculate FLOPS
    mFlops = mProp.multiProcessorCount * mProp.maxThreadsPerMultiProcessor * 2.0f;
}

MusaRuntime::~MusaRuntime() {}

bool MusaRuntime::isSupportedFP16() const {
    return mIsSupportedFP16;
}

bool MusaRuntime::isSupportedDotInt8() const {
    return mSupportDotInt8;
}

bool MusaRuntime::isSupportedDotAccInt8() const {
    return mSupportDotAccInt8;
}

std::vector<size_t> MusaRuntime::getMaxImage2DSize() {
    std::vector<size_t> result(2);
    result[0] = 16384; // Default max texture size
    result[1] = 16384;
    return result;
}

bool MusaRuntime::isCreateError() const {
    return mIsCreateError;
}

int MusaRuntime::device_id() const {
    return mDeviceId;
}

size_t MusaRuntime::mem_alignment_in_bytes() const {
    return 256;
}

void MusaRuntime::activate() {
    musaSetDevice(mDeviceId);
}

void* MusaRuntime::alloc(size_t size_in_bytes) {
    activate();
    void* ptr = nullptr;
    musaError_t err = musaMalloc(&ptr, size_in_bytes);
    if (err != musaSuccess) {
        MNN_ERROR("Failed to allocate MUSA memory: %zu bytes\n", size_in_bytes);
        return nullptr;
    }
    return ptr;
}

void MusaRuntime::free(void* ptr) {
    if (ptr != nullptr) {
        musaFree(ptr);
    }
}

void MusaRuntime::memcpy(void* dst, const void* src, size_t size_in_bytes, MNNMemcpyKind_t kind, bool sync) {
    musaMemcpyKind memcpyKind;
    switch (kind) {
        case MNNMemcpyHostToDevice:
            memcpyKind = musaMemcpyHostToDevice;
            break;
        case MNNMemcpyDeviceToHost:
            memcpyKind = musaMemcpyDeviceToHost;
            break;
        case MNNMemcpyDeviceToDevice:
            memcpyKind = musaMemcpyDeviceToDevice;
            break;
        default:
            return;
    }
    musaMemcpy(dst, src, size_in_bytes, memcpyKind);
    if (sync) {
        device_sync();
    }
}

void MusaRuntime::memset(void* dst, int value, size_t size_in_bytes) {
    musaMemset(dst, value, size_in_bytes);
}

void MusaRuntime::device_sync() {
    musaDeviceSynchronize();
}

size_t MusaRuntime::blocks_num(const size_t total_threads) {
    return (total_threads + mThreadPerBlock - 1) / mThreadPerBlock;
}

int MusaRuntime::selectDeviceMaxFreeMemory() {
    int deviceCount = 0;
    musaGetDeviceCount(&deviceCount);
    
    size_t maxFreeMemory = 0;
    int selectedDevice = 0;
    
    for (int i = 0; i < deviceCount; i++) {
        size_t freeMem, totalMem;
        musaMemGetInfo(&freeMem, &totalMem);
        if (freeMem > maxFreeMemory) {
            maxFreeMemory = freeMem;
            selectedDevice = i;
        }
    }
    return selectedDevice;
}

size_t MusaRuntime::getMemoryUsage(size_t size_in_bytes) const {
    return size_in_bytes;
}

std::pair<const void*, size_t> MusaRuntime::makeCache() {
    return std::make_pair(mCacheOutside, mCacheOutsideSize);
}

bool MusaRuntime::setCache(std::pair<const void*, size_t> cache) {
    mCacheOutside = cache.first;
    mCacheOutsideSize = cache.second;
    return true;
}

} // namespace MNN