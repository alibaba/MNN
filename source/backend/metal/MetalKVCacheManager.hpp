//
//  MetalKVCacheManager.hpp
//  MNN
//
//  Created by MNN on 2025/12/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef METAL_KVCACHE_MANAGER_HPP
#define METAL_KVCACHE_MANAGER_HPP

#import "core/Macro.h"
#import "core/MNNFileUtils.h"
#import "core/OpCommonUtils.hpp"
#import "core/KVCacheManager.hpp"

namespace MNN {

class MetalKVCacheManager : public KVCacheManager{
private:
    id<MTLBuffer> mKeyBuffer = nil;
    id<MTLBuffer> mValueBuffer = nil;
    id<MTLBuffer> mKScaleBuffer = nil;
    id<MTLBuffer> mVScaleBuffer = nil;
    // Only used when KV cache is stored on disk. For in-memory path V may use int8.
    size_t mCurrentTotalSize = 0;

    bool mQuantValue = false; // whether V is stored as int8 in cache
    bool mQuantKey = false;   // whether K is stored as int8 in cache
    std::shared_ptr<KVQuantParameter> mKVQuantParameter = nullptr;

private:
    void expandKVCacheInDisk(size_t oldSize, size_t curSize, size_t old_piece_stride, size_t old_piece_size, size_t new_piece_stride, bool need_copy, file_t specKeyFile = INVALID_FILE, file_t specValueFile = INVALID_FILE);
    bool expandKVCacheInMem(size_t oldSize, size_t old_piece_stride, size_t old_piece_size, size_t new_piece_stride, bool need_copy);
public:
    MetalKVCacheManager(Backend * backend, KVCacheConfig & kvConfig): KVCacheManager(backend, kvConfig) {
        // nothing todo
    }
    ~MetalKVCacheManager() {
        onClear();
    }
    Tensor * getKeyTensor() {
        return mPastKey.get();
    }
    Tensor * getValueTensor() {
        return mPastValue.get();
    }
    id<MTLBuffer> getKeyBuffer() {
        return mKeyBuffer;
    }
    id<MTLBuffer> getKScaleBuffer() {
        return mKScaleBuffer;
    }
    id<MTLBuffer> getVScaleBuffer() {
        return mVScaleBuffer;
    }
    id<MTLBuffer> getValueBuffer() {
        return mValueBuffer;
    }

    void setPastLength(int length) {
        mPastLength = length;
    }

    void setKVQuantParameter(std::shared_ptr<KVQuantParameter> p) {
        mKVQuantParameter = p;
    }
    void setAttenQuantKeyValue(bool quantKey, bool quantValue) {
        mQuantKey = quantKey;
        mQuantValue = quantValue;
    }
    bool useDynamicScaleBuffer() const {
        return (mQuantKey || mQuantValue) && mKVQuantParameter == nullptr;
    }
    bool quantValue() const {
        return mQuantValue;
    }

    virtual void onResize(int kv_num_head, int head_dim);
    virtual void onClear();
    virtual void onAlloc(KVMeta* meta, int seq_len);
    virtual void onRealloc(KVMeta* meta);
};

} // namespace MNN

#endif // METAL_KVCACHE_MANAGER_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
