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

class MetalExecution;

class MetalKVCacheManager : public KVCacheManager{
private:
    std::vector<MetalExecution*> mReplayUsers;
    void invalidateReplay();
    id<MTLBuffer> mKeyBuffer = nil;
    id<MTLBuffer> mValueBuffer = nil;
    id<MTLBuffer> mKScaleBuffer = nil;
    id<MTLBuffer> mVScaleBuffer = nil;
    // Only used when KV cache is stored on disk. K and V may have different element widths.
    size_t mCurrentKeySize = 0;
    size_t mCurrentValueSize = 0;

    bool mQuantValue = false; // whether V is stored as int8 in cache
    bool mQuantKey = false;   // whether K is stored as int8 in cache
    int mBatch = 1;
    std::shared_ptr<KVQuantParameter> mKVQuantParameter = nullptr;

private:
    ErrorCode expandKVCacheInDisk(size_t oldKeySize, size_t oldValueSize, size_t curKeySize, size_t curValueSize,
                                  int copy_len, bool need_copy, int oldMaxLength,
                                  file_t specKeyFile = INVALID_FILE, file_t specValueFile = INVALID_FILE);
    bool expandKVCacheInMem(size_t oldSize, int copy_len, bool need_copy, int oldMaxLength);
    ErrorCode growTo(int kvSeqLen, int pastLength);
public:
    void addReplayUser(MetalExecution* execution);
    void removeReplayUser(MetalExecution* execution);
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

    void setBatch(int batch) {
        if (batch > 0 && batch != mBatch) {
            onClear();
            mBatch = batch;
        }
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
    bool quantKey() const {
        return mQuantKey;
    }

    virtual void onResize(int kv_num_head, int head_dim);
    virtual void onClear();
    virtual ErrorCode onAlloc(KVMeta* meta, int seq_len);
    virtual ErrorCode onRealloc(KVMeta* meta);
    // Padded block decode forwards more query rows than meta->add commits, so the
    // cache must hold them even though the committed length does not grow.
    ErrorCode ensureCapacity(int kvSeqLen);
};

} // namespace MNN

#endif // METAL_KVCACHE_MANAGER_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
