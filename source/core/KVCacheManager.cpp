//
//  KVCacheManager.cpp
//  MNN
//
//  Created by MNN on 2024/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "KVCacheManager.hpp"
#include "core/Concurrency.h"

namespace MNN {

// Translate an address to a hex number string
static inline std::string addrToHex(void *addr) {
    std::string result = "";
    uint64_t n = (uint64_t)addr;
    for(int i = 15; i >= 0; i--) {
        int t = (n >> (i * 4)) & 0x0f;
        result.push_back((t < 10) ? ('0' + t) : ('A' + t - 10));
    }
    return result;
}

ErrorCode KVCacheManager::createKVCacheFile(std::string keyPath, std::string valuePath) {
    // Each layer has its own kvcache, so we have to create a key file and a value file for each layer and the file name must be unique
    // Here we use the address of the mResource as the file name because the addresses of mResource in different layers are guaranteed to be different
    std::string fileName = addrToHex(this);
    // Prefix-only sessions have no KV dir; fall back to the prefix dir, not the filesystem root.
    const std::string& dir = mConfig.mKVCacheDir.empty() ? mConfig.mPrefixCacheDir : mConfig.mKVCacheDir;
    mBaseFileName = MNNFilePathConcat(dir, fileName);

    mKeyCachePath   = keyPath.size() > 0 ? keyPath : mBaseFileName + ".k";
    mValueCachePath = valuePath.size() > 0 ? valuePath : mBaseFileName + ".v";
    mKeyCacheFD   = MNNCreateFile(mKeyCachePath.c_str());
    mValueCacheFD = MNNCreateFile(mValueCachePath.c_str());
    if (mKeyCacheFD == INVALID_FILE) {
        MNN_PRINT("Failed to create the file: %s\n", mKeyCachePath.c_str());
    }
    if (mValueCacheFD == INVALID_FILE) {
        MNN_PRINT("Failed to create the file: %s\n", mValueCachePath.c_str());
    }
    if (mKeyCacheFD == INVALID_FILE || mValueCacheFD == INVALID_FILE) {
        return FILE_CREATE_FAILED;
    }
    return NO_ERROR;
}

void KVCacheManager::closeKVCacheFile() {
    if (mKeyCacheFD != INVALID_FILE) {
        MNNCloseFile(mKeyCacheFD);
        mKeyCacheFD = INVALID_FILE;
    }
    if (mValueCacheFD != INVALID_FILE) {
        MNNCloseFile(mValueCacheFD);
        mValueCacheFD = INVALID_FILE;
    }
}

void KVCacheManager::removeKVCacheFile() {
    closeKVCacheFile();
    if (!mKeyCachePath.empty() && MNNRemoveFile(mKeyCachePath.c_str()) != MNN::NO_ERROR) {
        MNN_PRINT("Failed to remove the file: %s\n", mKeyCachePath.c_str());
    }
    if (!mValueCachePath.empty() && MNNRemoveFile(mValueCachePath.c_str()) != MNN::NO_ERROR) {
        MNN_PRINT("Failed to remove the file: %s\n", mValueCachePath.c_str());
    }
    mKeyCachePath.clear();
    mValueCachePath.clear();
}

ErrorCode KVCacheManager::resetKVCacheFileSize(size_t keySize, size_t valueSize) {
    if (MNNSetFileSize(mKeyCacheFD, keySize) != MNN::NO_ERROR || MNNSetFileSize(mValueCacheFD, valueSize) != MNN::NO_ERROR) {
        MNN_PRINT("Failed to resize the kvcache files!\n");
        return FILE_RESIZE_FAILED;
    }
    // A reported-success resize can still land short (quota, ENOSPC on the
    // deferred allocation), and the mmap would then be smaller than the rows
    // the shaders write. Verify the size we actually got.
    if (MNNGetFileSize(mKeyCacheFD) < keySize || MNNGetFileSize(mValueCacheFD) < valueSize) {
        MNN_ERROR("MNN::Metal: kv cache files shorter than requested (%zu/%zu bytes)\n", keySize, valueSize);
        return FILE_RESIZE_FAILED;
    }
    return NO_ERROR;
}

/*
**  @brief  Memory-map the kvcache file
**  @hint   After memory-mapping, we can access the kvcache files with pointers, just like accessing memory buffer
**          But the data actually resides in disk.
**          The OS will set some kernel page cache and manage the data swaping, which we do not need to care.
*/
ErrorCode KVCacheManager::mmapKVCache(size_t keySize, size_t valueSize, file_t specKeyFile, file_t specValueFile)
{
    // if keyFile or value file not given, use mKeyCacheFD or mValueCacheFD
    auto keyFrom = specKeyFile != INVALID_FILE ? specKeyFile : mKeyCacheFD;
    auto valueFrom = specValueFile != INVALID_FILE ? specValueFile : mValueCacheFD;

    if (mMapKeyAddr == nullptr) {
        mMapKeyAddr = (int8_t *)MNNMmapFile(keyFrom, keySize);
        if (mMapKeyAddr == nullptr) {
            MNN_PRINT("Failed to memory-map the kvcache!\n");
        }
    }

    if (mMapValueAddr == nullptr) {
        mMapValueAddr = (int8_t *)MNNMmapFile(valueFrom, valueSize);
        if (mMapValueAddr == nullptr) {
            MNN_PRINT("Failed to memory-map the kvcache!\n");
        }
    }
    if (mMapKeyAddr == nullptr || mMapValueAddr == nullptr) {
        // A half-mapped pair must not survive: the caller would index the null
        // side, and a later mmap would skip it because the address looks live.
        unmapKVCache(keySize, valueSize);
        return OUT_OF_MEMORY;
    }
    return NO_ERROR;
}

void KVCacheManager::unmapKVCache(size_t keySize, size_t valueSize)
{
    if (mMapKeyAddr != nullptr) {
        MNNMmapSync(mMapKeyAddr, keySize);
        MNNUnmapFile(mMapKeyAddr, keySize);
        mMapKeyAddr = nullptr;
    }
    if (mMapValueAddr != nullptr) {
        MNNMmapSync(mMapValueAddr, valueSize);
        MNNUnmapFile(mMapValueAddr, valueSize);
        mMapValueAddr = nullptr;
    }
}

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
