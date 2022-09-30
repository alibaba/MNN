//
//  CUDARuntime.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cuda/core/runtime/CUDARuntime.hpp"
#include <sys/stat.h>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "core/Macro.h"

//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

namespace MNN {

bool CUDARuntime::isCreateError() const {
    return mIsCreateError;
}

CUDARuntime::CUDARuntime(int device_id) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start CUDARuntime id:%d\n", device_id);
#endif
    int version;
    cuda_check(cudaRuntimeGetVersion(&version));
    int id = device_id;
    if (id < 0) {
        cuda_check(cudaGetDevice(&id));
    }
    // printf("use GPU device id:%d\n", id);
    // id = selectDeviceMaxFreeMemory();
    // cuda_check(cudaSetDevice(id));

    mDeviceId = id;
    cuda_check(cudaGetDeviceProperties(&mProp, id));
    MNN_ASSERT(mProp.maxThreadsPerBlock > 0);
#ifdef MNN_CUDA_USE_BLAS
    cublas_check(cublasCreate(&mCublasHandle));
    cublas_check(cublasSetPointerMode(mCublasHandle, CUBLAS_POINTER_MODE_HOST));
#endif
}

CUDARuntime::~CUDARuntime() {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ~CUDARuntime !\n");
#endif
#ifdef MNN_CUDA_USE_BLAS
    cublas_check(cublasDestroy(mCublasHandle));
#endif
#ifdef LOG_VERBOSE
    MNN_PRINT("end ~CUDARuntime !\n");
#endif
}

int CUDARuntime::selectDeviceMaxFreeMemory() {
    cudaDeviceProp deviceProp;
    int deviceCount;
    cuda_check(cudaGetDeviceCount(&deviceCount));

    // Check id:0 card info
    int id = 0;
    cuda_check(cudaSetDevice(0));
    size_t total_size = 0, free_size_max = 0;
    cudaError_t memStatus = cudaMemGetInfo(&free_size_max, &total_size);
    cuda_check(memStatus);
    // printf("card:0, free:%zu, total:%zu, memStatusSuccess:%d\n", free_size_max, total_size, memStatus == cudaSuccess);

    for(int i = 1; i < deviceCount; i++) {
        cuda_check(cudaSetDevice(i));
        size_t free_size;
        cuda_check(cudaMemGetInfo(&free_size, &total_size));
        if(free_size > free_size_max) {
            free_size_max = free_size;
            id = i;
        }
        // printf("card:%d, free:%zu, total:%zu\n", i, free_size, total_size);
    }
    return id;
}

size_t CUDARuntime::blocks_num(const size_t total_threads) {
    // size_t maxNum = mProp.maxThreadsPerBlock;
    // if(total_threads / 32 > maxNum) {
    //     mThreadPerBlock = maxNum;
    // } else if(total_threads / 16 > maxNum) {
    //     mThreadPerBlock = maxNum / 2;
    // } else if(total_threads / 8 > maxNum) {
    //     mThreadPerBlock = maxNum / 4;
    // } else if(total_threads / 4 > maxNum) {
    //     mThreadPerBlock = maxNum / 8;
    // } else {
    //     mThreadPerBlock = 128;
    // }

    mThreadPerBlock = 128;
    return (total_threads + mThreadPerBlock - 1) / mThreadPerBlock;
}

bool CUDARuntime::isSupportedFP16() const {
    return mIsSupportedFP16;
}

bool CUDARuntime::isSupportedDotInt8() const {
    return mSupportDotInt8;
}

bool CUDARuntime::isSupportedDotAccInt8() const {
    return mSupportDotAccInt8;
}

size_t CUDARuntime::mem_alignment_in_bytes() const {
    return std::max(mProp.textureAlignment, mProp.texturePitchAlignment);
}

int CUDARuntime::device_id() const {
    return mDeviceId;
}

void CUDARuntime::activate() {
    int id = device_id();
    if (id >= 0) {
        cuda_check(cudaSetDevice(id));
    }
}

void *CUDARuntime::alloc(size_t size_in_bytes) {
    void *ptr = nullptr;
    cuda_check(cudaMalloc(&ptr, size_in_bytes));
    MNN_ASSERT(nullptr != ptr);
    checkKernelErrors;
    return ptr;
}

void CUDARuntime::free(void *ptr) {
    cuda_check(cudaFree(ptr));
}

void CUDARuntime::memcpy(void *dst, const void *src, size_t size_in_bytes, MNNMemcpyKind_t kind, bool sync) {
    cudaMemcpyKind cuda_kind;
    switch (kind) {
        case MNNMemcpyDeviceToHost:
            cuda_kind = cudaMemcpyDeviceToHost;
            break;
        case MNNMemcpyHostToDevice:
            cuda_kind = cudaMemcpyHostToDevice;
            break;
        case MNNMemcpyDeviceToDevice:
            cuda_kind = cudaMemcpyDeviceToDevice;
            break;
        default:
            MNN_ERROR("bad cuda memcpy kind\n");
    }
    //TODO, support Async Afterwards
    cuda_check(cudaMemcpy(dst, src, size_in_bytes, cuda_kind));
    checkKernelErrors;
}

void CUDARuntime::memset(void *dst, int value, size_t size_in_bytes) {
    cuda_check(cudaMemset(dst, value, size_in_bytes));
    checkKernelErrors;
}



} // namespace MNN
