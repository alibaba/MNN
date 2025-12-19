//
//  CPUKVCacheManager.hpp
//  MNN
//
//  Created by MNN on 2024/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef CPU_KVCACHE_MANAGER_HPP
#define CPU_KVCACHE_MANAGER_HPP

#include "core/KVCacheManager.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#if defined (__aarch64__)
#define FLOAT16_T __fp16
#else
#define FLOAT16_T float
#endif

typedef uint8_t fp8_t;

#define QUANT_INFO_BYTES 4

namespace MNN {

class CPUKVCacheManager : public KVCacheManager{
private:
    int  eP, lP, hP;                                // Packing mode for float matmul
    int  eP8, lP8, hP8;                             // Packing mode for int8 gemm kernel
    int  mThreadNum = 1;
    
    size_t mFlashAttentionUpperKv = 0;
    
    void expandKVCacheInMem(int oldMaxLength);
    void moveKVCacheFromMemToDisk(int oldMaxLength);
    void expandKVCacheInDisk(int oldMaxLength, int oldKeySize, int oldValueSize, int keySize, int valueSize, file_t specKeyFile = INVALID_FILE, file_t specValueFile = INVALID_FILE);
    template <typename T> void ProcessKey(const Tensor* key, int seq_len, int kv_h);
    template <typename T> void ProcessValue(const Tensor* value, int seq_len, int kv_h);
    template <typename T> void moveKV(int src, int dst, int size);
    size_t keyIndex(int seq, int dim) const;
    size_t valueIndex(int seq, int dim) const;
    void saveKVCacheInDisk();

    // The key/value size must be updated on every alloc or realloc call.
    size_t mCurrentKeySizePerHead = 0;
    size_t mCurrentValueSizePerHead = 0;

    // flash attention
    bool mUseFlashAttention = true;

    // quant Key/Value
    bool mQuantValue    = false;                    // Quantize values to int8 or not
    bool mQuantKey      = false;                    // Whether to use int8 gemm kernel in CPU attention
    std::shared_ptr<Tensor> mKeySum;                // numhead, [maxlen/hP8, hP8]
    std::shared_ptr<Tensor> mValueSum;              // numhead, [headDim/hP8, hP8]
    std::shared_ptr<Tensor> mKeyMax;                // {numhead, headDim}
    decltype(CoreFunctions::MNNQuantAttentionKey) mQuantKeyFunc;
    decltype(CoreFunctions::MNNQuantAttentionValue) mQuantValueFunc;
public:
    CPUKVCacheManager(Backend * backend, KVCacheConfig & kvConfig): KVCacheManager(backend, kvConfig) {
        // nothing todo
    }
    ~CPUKVCacheManager() {
        onClear();
    }
    const Tensor * keySum() {
        return mKeySum.get();
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
        return baseAddr + kv_h * mCurrentKeySizePerHead;
    }
    int8_t * addrOfValue(int kv_h) {
        int8_t * baseAddr = mKVCacheInDisk ? mMapValueAddr : mPastValue->host<int8_t>();
        return baseAddr + kv_h * mCurrentValueSizePerHead;

    }
    void setFlashAttentionUpperKv(size_t upperKv) {
        mFlashAttentionUpperKv = upperKv;
    }
    size_t getFlashAttentionBlockKv() {
        return mFlashAttentionUpperKv;
    }
    
    void onPushBack(const Tensor * key, const Tensor * value, int add);
    void onDequantValue(Tensor * dequantedValues);
    void onUpdateKV(const Tensor * key, const Tensor * value, int add);

    // quant Key/Value
    int8_t * addrOfKeySum(int kv_h) {
        if (mQuantKey) {
            return mKeySum->host<int8_t>() + kv_h * UP_DIV(mMaxLength, hP8) * hP8 * 4;
        }else {
            return nullptr;
        }
    }
    int8_t* addrOfKeyMax(int kvH) {
        if (mQuantKey) {
            return mKeyMax->host<int8_t>() + kvH * mHeadDim * mBytes;
        } else {
            return nullptr;
        }
    }
    int8_t* addrOfValueSum(int kvH) {
        if (mQuantValue) {
            return mValueSum->host<int8_t>() + kvH * mValueSum->stride(0);
        } else {
            return nullptr;
        }
    }
    void setAttenQuantKeyValue(bool useFlashAttention, bool quantKey, bool quantValue) {
        mUseFlashAttention = useFlashAttention;
        mQuantValue = quantValue;
        mQuantKey = quantKey;
    }

    virtual void onResize(int kv_num_head, int head_dim);
    virtual void onClear();
    virtual void onAlloc(KVMeta* meta, int seq_len);
    virtual void onRealloc(KVMeta* meta);

};

} // namespace MNN

#endif // CPU_KVCACHE_MANAGER_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
