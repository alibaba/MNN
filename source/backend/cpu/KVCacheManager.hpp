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
        bool mQuantKey   = false;               // Quantize keys to int8 or not
        bool mQuantValue = false;               // Quantize values to fp8 or not
        std::string mKVCacheDir = "/tmp";       // Path of the kvcache files in disk
        size_t mKVCacheSizeLimit = -1;          // The limit of the kvcache size
        int  mExpandChunk = 64;                 // Number of expand chunks when the buffer is full
    };
private:
    Backend * mBackend;
    KVCacheConfig mConfig;
    std::shared_ptr<Tensor> mPastKey;               // numhead, [maxlen/eP, headdim, eP]
    std::shared_ptr<Tensor> mPastValue;             // numhead, [headdim/eP, maxlen, eP]
    std::shared_ptr<Tensor> mDequantKeyScale;       // numhead, [maxlen/eP, 1, eP]
    std::shared_ptr<Tensor> mDequantKeyZeroPoint;   // numhead, [maxlen/eP, 1, eP]
    file_t mKeyCacheFD   = INVALID_FILE;            // The file descriptor of keys
    file_t mValueCacheFD = INVALID_FILE;            // The file descriptor of values
    char * mMapKeyAddr   = nullptr;                 // Memory-mapped address of keys
    char * mMapValueAddr = nullptr;                 // Memory-mapped address of values
    bool mKVCacheInDisk  = false;                   // Whether the kvcache is in disk or in memory now
    int  mPastLength     = 0;                       // Length of past kvcache
    int  mMaxLength      = 0;                       // Capacity of current kvcache buffer (how many kv items can be stored at most)
    int  eP, lP, hP, mBytes, mThreadNum;
    int  mKvNumHead = 0, mHeadDim   = 0;
    void createKVCacheFile();
    void removeKVCacheFile();
    void resetKVCacheFileSize(size_t keySize, size_t valueSize);
    void mmapKVCache(size_t keySize, size_t valueSize);
    void unmapKVCache(size_t keySize, size_t valueSize);
    void expandKVCacheInMem(int oldMaxLength);
    void moveKVCacheFromMemToDisk(int oldMaxLength);
    void expandKVCacheInDisk(int oldMaxLength);
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
        return mDequantKeyScale.get();
    }
    const Tensor * zeroPoint() {
        return mDequantKeyZeroPoint.get();
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
    char * addrOfKey(int kv_h) {
        char * baseAddr = mKVCacheInDisk ? mMapKeyAddr : mPastKey->host<char>();
        return baseAddr + kv_h * UP_DIV(mMaxLength, hP) * mHeadDim * hP * (mConfig.mQuantKey ? 1 : mBytes);
    }
    char * addrOfValue(int kv_h) {
        char * baseAddr = mKVCacheInDisk ? mMapValueAddr : mPastValue->host<char>();
        return baseAddr + kv_h * UP_DIV(mHeadDim, hP) * mMaxLength * hP * (mConfig.mQuantValue ? 1 : mBytes);
    }
    char * addrOfScale(int kv_h) {
        if (mConfig.mQuantKey == false)
            return nullptr;
        char * baseAddr = mDequantKeyScale->host<char>();
        return baseAddr + kv_h * UP_DIV(mMaxLength, hP) * 1 * hP * mBytes;
    }
    char * addrOfZeroPoint(int kv_h) {
        if (mConfig.mQuantKey == false)
            return nullptr;
        char * baseAddr = mDequantKeyZeroPoint->host<char>();
        return baseAddr + kv_h * UP_DIV(mMaxLength, hP) * 1 * hP * mBytes;
    }
    void onResize(int kv_num_head, int head_dim);
    void onAlloc(int kv_seq_len);
    void onRealloc(int kv_seq_len);
    void onClear();
    void onPushBack(const Tensor * key, const Tensor * value);
    void onDequantValue(Tensor * dequantedValues);
};

} // namespace MNN

#endif // KVCACHE_MANAGER_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE