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
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
// #define LOG_VERBOSE
#define CUDNN_VERSION_STR STR(CUDNN_MAJOR) "." STR(CUDNN_MINOR) "." STR(CUDNN_PATCHLEVEL)

#pragma message "compile with cuda " STR(CUDART_VERSION) " "
#pragma message "compile with cuDNN " CUDNN_VERSION_STR " "

static_assert(!(CUDNN_MAJOR == 5 && CUDNN_MINOR == 1), "cuDNN 5.1.x series has bugs. Use 5.0.x instead.");

#undef STR
#undef STR_HELPER

namespace MNN {

bool CUDARuntime::isCreateError() const {
    return mIsCreateError;
}

CUDARuntime::CUDARuntime(bool permitFloat16, int device_id) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start CUDARuntime !\n");
#endif
    int version;
    cuda_check(cudaRuntimeGetVersion(&version));
    int id = device_id;
    if (id < 0) {
        cuda_check(cudaGetDevice(&id));
    }
    mDeviceId = id;
    cuda_check(cudaGetDeviceProperties(&mProp, id));
    MNN_ASSERT(mProp.maxThreadsPerBlock > 0);

    cublas_check(cublasCreate(&mCublasHandle));

    // Set stream for cuDNN and cublas handles.

    // Note that all cublas scalars (alpha, beta) and scalar results such as dot
    // output resides at device side.
    cublas_check(cublasSetPointerMode(mCublasHandle, CUBLAS_POINTER_MODE_HOST));
    cudnn_check(cudnnCreate(&mCudnnHandle));
}

CUDARuntime::~CUDARuntime() {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ~CUDARuntime !\n");
#endif
    cublas_check(cublasDestroy(mCublasHandle));
    cudnn_check(cudnnDestroy(mCudnnHandle));

#ifdef LOG_VERBOSE
    MNN_PRINT("end ~CUDARuntime !\n");
#endif
}

int CUDARuntime::blocks_num(const int total_threads) {
    int maxNum = mProp.maxThreadsPerBlock;
    if(total_threads / 32 > maxNum) {
        mThreadPerBlock = maxNum;
    } else if(total_threads / 16 > maxNum) {
        mThreadPerBlock = maxNum / 2;
    } else if(total_threads / 8 > maxNum) {
        mThreadPerBlock = maxNum / 4;
    } else if(total_threads / 4 > maxNum) {
        mThreadPerBlock = maxNum / 8;
    } else {
        mThreadPerBlock = 128;
    }
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
}

void CUDARuntime::memset(void *dst, int value, size_t size_in_bytes) {
    cuda_check(cudaMemset(dst, value, size_in_bytes));
}

cublasHandle_t CUDARuntime::cublas_handle() {
    return mCublasHandle;
}

cudnnHandle_t CUDARuntime::cudnn_handle() {
    return mCudnnHandle;
}

} // namespace MNN
