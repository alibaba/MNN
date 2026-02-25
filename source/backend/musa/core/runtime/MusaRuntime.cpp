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
    
    musaError_t err = musaSetDevice(mDeviceId);
    if (err != musaSuccess) {
        MNN_ERROR("Failed to set MUSA device %d\n", mDeviceId);
        mIsCreateError = true;
        return;
    }
    
    err = musaGetDeviceProperties(&mProp, mDeviceId);
    if (err != musaSuccess) {
        MNN_ERROR("Failed to get MUSA device properties\n");
        mIsCreateError = true;
        return;
    }
    
    // Check FP16 support
    mIsSupportedFP16 = true; // Assume FP16 support for Moore Threads GPUs
    
    // Calculate FLOPS
    mFlops = mProp.multiProcessorCount * mProp.maxThreadsPerMultiProcessor * 2.0f;
    
    MNN_PRINT("MUSA Device: %s\n", mProp.name);
    MNN_PRINT("MUSA Compute Capability: %d.%d\n", mProp.major, mProp.minor);
    MNN_PRINT("MUSA Multiprocessor Count: %d\n", mProp.multiProcessorCount);
    MNN_PRINT("MUSA Shared Memory Per Block: %d bytes\n", mProp.sharedMemPerBlock);
}

MusaRuntime::~MusaRuntime() {
    // Cleanup if needed
}

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
    result[0] = mProp.maxTexture2D[0];
    result[1] = mProp.maxTexture2D[1];
    return result;
}

bool MusaRuntime::isCreateError() const {
    return mIsCreateError;
}

int MusaRuntime::device_id() const {
    return mDeviceId;
}

size_t MusaRuntime::mem_alignment_in_bytes() const {
    return 256; // Default alignment for MUSA
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
    activate();
    if (ptr != nullptr) {
        musaFree(ptr);
    }
}

void MusaRuntime::memcpy(void* dst, const void* src, size_t size_in_bytes, MNNMemcpyKind_t kind, bool sync) {
    activate();
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
            MNN_ERROR("Unknown memcpy kind\n");
            return;
    }
    
    musaError_t err = musaMemcpy(dst, src, size_in_bytes, memcpyKind);
    if (err != musaSuccess) {
        MNN_ERROR("MUSA memcpy failed\n");
    }
    
    if (sync) {
        device_sync();
    }
}

void MusaRuntime::memset(void* dst, int value, size_t size_in_bytes) {
    activate();
    musaMemset(dst, value, size_in_bytes);
}

void MusaRuntime::device_sync() {
    activate();
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

size_t MusaRuntime::getMemoryUsage(const Tensor* tensor) const {
    return tensor->size();
}

std::pair<const void*, size_t> MusaRuntime::makeCache() {
    // Cache implementation for MUSA
    return std::make_pair(mCacheOutside, mCacheOutsideSize);
}

bool MusaRuntime::setCache(std::pair<const void*, size_t> cache) {
    mCacheOutside = cache.first;
    mCacheOutsideSize = cache.second;
    return true;
}

} // namespace MNN
