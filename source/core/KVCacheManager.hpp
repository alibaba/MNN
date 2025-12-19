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


namespace MNN {

class KVCacheManager : public NonCopyable{
public:
    struct KVCacheConfig {
        std::string mKVCacheDir;       // Path of the kvcache files in disk
        std::string mPrefixCacheDir;   // Path of the prefix prompt kvcache files in disk
        int  mExpandChunk = 64;                 // Number of expand chunks when the buffer is full
        int mBlockNum = 1;
        int mKvAlignNum;
    };
protected:
    Backend * mBackend;
    KVCacheConfig mConfig;
    std::shared_ptr<Tensor> mPastKey;               // {numhead, [maxlen/hP, headdim, hP]} or {numhead, [maxlen/hP8, headdim/lP8, hP8, lP8]}
    std::shared_ptr<Tensor> mPastValue;             // numhead, [headdim/hP, maxlen, hP]
    file_t mKeyCacheFD   = INVALID_FILE;            // The file descriptor of keys
    file_t mValueCacheFD = INVALID_FILE;            // The file descriptor of values
    int8_t * mMapKeyAddr   = nullptr;                 // Memory-mapped address of keys
    int8_t * mMapValueAddr = nullptr;                 // Memory-mapped address of values
    bool mKVCacheInDisk  = false;                   // Whether the kvcache is in disk or in memory now
    bool mSaveShareKvPrefix  = false;
    int  mPastLength     = 0;                       // Length of past kvcache
    int  mMaxLength      = 0;                       // Capacity of current kvcache buffer (how many kv items can be stored at most)
    int  mBytes = 4;
    int  mKvNumHead = 0, mHeadDim = 0;
    KVMeta* mMeta;
    std::string mBaseFileName;
    std::string mBasePrefixFileName;
    
    void createKVCacheFile(std::string keyPath = "", std::string valuePath = "");
    void removeKVCacheFile();
    void resetKVCacheFileSize(size_t keySize, size_t valueSize);
    void mmapKVCache(size_t keySize, size_t valueSize, file_t specKeyFile = INVALID_FILE, file_t specValueFile = INVALID_FILE);
    void unmapKVCache(size_t keySize, size_t valueSize);
    
public:
    KVCacheManager(Backend * backend, KVCacheConfig & kvConfig) {
        mBackend   = backend;
        mConfig    = kvConfig;
    }
    ~KVCacheManager() {
        // nothing todo
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

    bool inDisk() {
        return mKVCacheInDisk;
    }
    int kvLength() {
        return mPastLength;
    }
    int maxLength() {
        return mMaxLength;
    }

    virtual void onResize(int kv_num_head, int head_dim) = 0;
    virtual void onClear() = 0;
    virtual void onAlloc(KVMeta* meta, int seq_len) = 0;
    virtual void onRealloc(KVMeta* meta) = 0;
};

} // namespace MNN

#endif // KVCACHE_MANAGER_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
