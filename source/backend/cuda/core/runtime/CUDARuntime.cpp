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
#include "execution/cutlass_common/tune/CudaCache_generated.h"

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
    cuda_check(cudaGetDeviceCount(&mDeviceCount));
    if (id < 0 || id >= mDeviceCount) {
        cuda_check(cudaGetDevice(&id));
    }
    
    // printf("use GPU device id:%d\n", id);
    // id = selectDeviceMaxFreeMemory();
    cuda_check(cudaSetDevice(id));

    mDeviceId = id;
    cuda_check(cudaGetDeviceProperties(&mProp, id));
    MNN_ASSERT(mProp.maxThreadsPerBlock > 0);
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

    // Check id:0 card info
    int id = 0;
    cuda_check(cudaSetDevice(0));
    size_t total_size = 0, free_size_max = 0;
    cudaError_t memStatus = cudaMemGetInfo(&free_size_max, &total_size);
    cuda_check(memStatus);
    // printf("card:0, free:%zu, total:%zu, memStatusSuccess:%d\n", free_size_max, total_size, memStatus == cudaSuccess);

    for(int i = 1; i < mDeviceCount; i++) {
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

void CUDARuntime::device_sync() {
    cuda_check(cudaDeviceSynchronize());
}

std::pair<const void*, size_t> CUDARuntime::makeCache() {
    std::unique_ptr<CudaCache::CacheT> cache(new CudaCache::CacheT);

    for (auto& iter : mTunedBlockWarpShape) {
        std::unique_ptr<CudaCache::AutotuningT> tuning(new CudaCache::AutotuningT);
        tuning->params = iter.first.first;
        tuning->problemSize = iter.first.second;
        
        tuning->threadBlockSize = iter.second.first;
        tuning->timeCost = iter.second.second;

        cache->tunings.emplace_back(std::move(tuning));
    }

    flatbuffers::FlatBufferBuilder builder;
    auto lastOffset = CudaCache::Cache::Pack(builder, cache.get());
    builder.Finish(lastOffset);
    mBuffer.resize(builder.GetSize());
    ::memcpy(mBuffer.data(), builder.GetBufferPointer(), builder.GetSize());
    return std::make_pair(mBuffer.data(), mBuffer.size());
}

bool CUDARuntime::setCache(std::pair<const void*, size_t> cache) {
    auto buffer = cache.first;
    auto size = cache.second;
    if (nullptr == buffer) {
        mCacheOutside = nullptr;
        mCacheOutsideSize = 0;
        mBuffer.clear();
        return false;//actually get nothing
    }
    mCacheOutsideSize = size;
    mCacheOutside = buffer;
    auto cacheBuffer = CudaCache::GetCache(buffer);
    flatbuffers::Verifier verify((const uint8_t*)buffer, size);
    if (false == CudaCache::VerifyCacheBuffer(verify)) {
        return false;
    }
    if (nullptr == cacheBuffer->tunings()) {
        return false;
    }

    // Load Auto Tuning Info
    if (nullptr != cacheBuffer->tunings()) {
        auto tuningInfo = cacheBuffer->tunings();
        for (int i=0; i<tuningInfo->size(); ++i) {
            auto tun = tuningInfo->GetAs<CudaCache::Autotuning>(i);
            if (nullptr == tun->params() || nullptr == tun->problemSize()) {
                MNN_ERROR("Error tunning info\n");
                continue;
            }
            std::vector<int32_t> param(tun->params()->size());
            for (int v=0; v<param.size(); ++v) {
                param[v] = tun->params()->data()[v];
            }
            std::vector<uint32_t> problem(tun->problemSize()->size());
            for (int v=0; v<problem.size(); ++v) {
                problem[v] = tun->problemSize()->data()[v];
            }
            std::string blockShape = tun->threadBlockSize()->str();
            uint32_t cost = tun->timeCost();
            mTunedBlockWarpShape.insert(std::make_pair(std::make_pair(param, problem), std::make_pair(blockShape, cost)));
        }
    }
    return true;
}

} // namespace MNN
