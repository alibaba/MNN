//
//  KVCacheManager.hpp
//  MNN
//
//  Created by MNN on 2024/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef KVCACHE_MANAGER_HPP
#define KVCACHE_MANAGER_HPP

#include "core/Macro.h"
#include "core/MNNFileUtils.h"
#include "core/OpCommonUtils.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#if defined (__aarch64__)
#define FLOAT16_T __fp16
#else
#define FLOAT16_T float
#endif

typedef uint8_t fp8_t;

namespace MNN {

class KVCacheManager : public NonCopyable{
public:
    struct KVCacheConfig {
        bool mQuantKey      = false;            // Quantize keys to int8 or not
        bool mQuantValue    = false;            // Quantize values to fp8 or not
        bool mUseInt8Kernel = false;            // Whether to use int8 gemm kernel in CPU attention
        std::string mKVCacheDir = "/tmp";       // Path of the kvcache files in disk
        size_t mKVCacheSizeLimit = -1;          // The limit of the kvcache size
        int  mExpandChunk = 64;                 // Number of expand chunks when the buffer is full
    };
private:
    Backend * mBackend;
    KVCacheConfig mConfig;
    std::shared_ptr<Tensor> mPastKey;               // {numhead, [maxlen/hP, headdim, hP]} or {numhead, [maxlen/hP8, headdim/lP8, hP8, lP8]}
    std::shared_ptr<Tensor> mPastValue;             // numhead, [headdim/hP, maxlen, hP]
    std::shared_ptr<Tensor> mKeyScale;              // {numhead, [maxlen/hP, hP]} or {numhead, [maxlen/hP8, hP8]}
    std::shared_ptr<Tensor> mKeyZeroPoint;          // {numhead, [maxlen/hP, hP]} or {numhead, [maxlen/hP8, hP8]}
    std::shared_ptr<Tensor> mKeySum;                // numhead, [maxlen/hP8, hP8]
    file_t mKeyCacheFD   = INVALID_FILE;            // The file descriptor of keys
    file_t mValueCacheFD = INVALID_FILE;            // The file descriptor of values
    int8_t * mMapKeyAddr   = nullptr;                 // Memory-mapped address of keys
    int8_t * mMapValueAddr = nullptr;                 // Memory-mapped address of values
    bool mKVCacheInDisk  = false;                   // Whether the kvcache is in disk or in memory now
    int  mPastLength     = 0;                       // Length of past kvcache
    int  mMaxLength      = 0;                       // Capacity of current kvcache buffer (how many kv items can be stored at most)
    int  eP, lP, hP;                                // Packing mode for float matmul
    int  eP8, lP8, hP8;                             // Packing mode for int8 gemm kernel
    int  mBytes = 4, mThreadNum = 1;
    int  mKvNumHead = 0, mHeadDim = 0;
    void createKVCacheFile();
    void removeKVCacheFile();
    void resetKVCacheFileSize(size_t keySize, size_t valueSize);
    void mmapKVCache(size_t keySize, size_t valueSize);
    void unmapKVCache(size_t keySize, size_t valueSize);
    void expandKVCacheInMem(int oldMaxLength);
    void moveKVCacheFromMemToDisk(int oldMaxLength);
    void expandKVCacheInDisk(int oldMaxLength, int oldKeySize, int oldValueSize, int keySize, int valueSize);
    template <typename T> void pack_key(const Tensor* key, int seq_len, int kv_h);
    template <typename T> void pack_value(const Tensor* value, int seq_len, int kv_h);
    template <typename T> void moveKV(int src, int dst, int size);
    size_t keyIndex(int seq, int dim) const;
    size_t valueIndex(int seq, int dim) const;
public:
    KVCacheManager(Backend * backend, KVCacheConfig & kvConfig) {
        mBackend   = backend;
        mConfig    = kvConfig;
    }
    ~KVCacheManager() {
        onClear();
    }
    const Backend * backend() {
        return mBackend;
    }
    const KVCacheConfig * config() {
        return &mConfig;
    }
    const Tensor * key() {
        return mPastKey.get();
    }
    const Tensor * value() {
        return mPastValue.get();
    }
    const Tensor * scale() {
        return mKeyScale.get();
    }
    const Tensor * zeroPoint() {
        return mKeyZeroPoint.get();
    }
    const Tensor * keySum() {
        return mKeySum.get();
    }
    bool inDisk() {
        return mKVCacheInDisk;
    }
    int kvLength() {
        return mPastLength;
    }
    int maxLength() {
        return mMaxLength;
    }
    uint8_t* keyAddr() {
        int8_t * baseAddr = mKVCacheInDisk ? mMapKeyAddr : mPastKey->host<int8_t>();
        return (uint8_t*)baseAddr;
    }
    uint8_t* valudAddr() {
        int8_t * baseAddr = mKVCacheInDisk ? mMapValueAddr : mPastValue->host<int8_t>();
        return (uint8_t*)baseAddr;
    }
    int8_t * addrOfKey(int kv_h) {
        int8_t * baseAddr = mKVCacheInDisk ? mMapKeyAddr : mPastKey->host<int8_t>();
        if (mConfig.mUseInt8Kernel) {
            return baseAddr + kv_h * UP_DIV(mMaxLength, hP8) * UP_DIV(mHeadDim, lP8) * hP8 * lP8;
        } else if (mConfig.mQuantKey) {
            return baseAddr + kv_h * UP_DIV(mMaxLength, hP) * ROUND_UP(mHeadDim, lP) * hP;
        } else {
            return baseAddr + kv_h * UP_DIV(mMaxLength, hP) * ROUND_UP(mHeadDim, lP) * hP * mBytes;
        }
    }
    int8_t * addrOfValue(int kv_h) {
        int8_t * baseAddr = mKVCacheInDisk ? mMapValueAddr : mPastValue->host<int8_t>();
        if (mConfig.mQuantValue) {
            return baseAddr + kv_h * UP_DIV(mHeadDim, hP) * ROUND_UP(mMaxLength, lP) * hP;
        } else {
            return baseAddr + kv_h * UP_DIV(mHeadDim, hP) * ROUND_UP(mMaxLength, lP) * hP * mBytes;
        }
    }
    int8_t * addrOfScale(int kv_h) {
        if (mConfig.mUseInt8Kernel) {
            return mKeyScale->host<int8_t>() + kv_h * UP_DIV(mMaxLength, hP8) * hP8 * 4;
        } else if (mConfig.mQuantKey) {
            return mKeyScale->host<int8_t>() + kv_h * UP_DIV(mMaxLength, hP) * hP * mBytes;
        } else {
            return nullptr;
        }
    }
    int8_t * addrOfZeroPoint(int kv_h) {
        if (mConfig.mUseInt8Kernel) {
            return mKeyZeroPoint->host<int8_t>() + kv_h * UP_DIV(mMaxLength, hP8) * hP8 * 4;
        } else if (mConfig.mQuantKey) {
            return mKeyZeroPoint->host<int8_t>() + kv_h * UP_DIV(mMaxLength, hP) * hP * mBytes;
        } else {
            return nullptr;
        }
    }
    int8_t * addrOfKeySum(int kv_h) {
        if (mConfig.mUseInt8Kernel) {
            return mKeySum->host<int8_t>() + kv_h * UP_DIV(mMaxLength, hP8) * hP8 * 4;
        }else {
            return nullptr;
        }
    }
    void onResize(int kv_num_head, int head_dim);
    void onAlloc(int kv_seq_len);
    void onRealloc(const KVMeta* meta);
    void onClear();
    void onPushBack(const Tensor * key, const Tensor * value, int add);
    void onDequantValue(Tensor * dequantedValues);
};

} // namespace MNN

#endif // KVCACHE_MANAGER_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
