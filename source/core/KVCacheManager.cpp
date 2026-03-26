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

void KVCacheManager::createKVCacheFile(std::string keyPath, std::string valuePath) {
    // Each layer has its own kvcache, so we have to create a key file and a value file for each layer and the file name must be unique
    // Here we use the address of the mResource as the file name because the addresses of mResource in different layers are guaranteed to be different
    std::string fileName = addrToHex(this);
    mBaseFileName = MNNFilePathConcat(mConfig.mKVCacheDir, fileName);

    std::string pathk    = keyPath.size() > 0 ? keyPath : mBaseFileName + ".k";
    std::string pathv    = valuePath.size() > 0 ? valuePath : mBaseFileName + ".v";
    mKeyCacheFD   = MNNCreateFile(pathk.c_str());
    mValueCacheFD = MNNCreateFile(pathv.c_str());
    if (mKeyCacheFD == INVALID_FILE) {
        MNN_PRINT("Failed to create the file: %s\n", pathk.c_str());
    }
    if (mValueCacheFD == INVALID_FILE) {
        MNN_PRINT("Failed to create the file: %s\n", pathv.c_str());
    }
}

void KVCacheManager::removeKVCacheFile() {
    std::string pathk    = mBaseFileName + ".k";
    std::string pathv    = mBaseFileName + ".v";
    if (mKeyCacheFD != INVALID_FILE) {
        MNNCloseFile(mKeyCacheFD);
        mKeyCacheFD = INVALID_FILE;
        if (MNNRemoveFile(pathk.c_str()) != MNN::NO_ERROR) {
            MNN_PRINT("Failed to remove the file: %s\n", pathk.c_str());
        }
    }
    if (mValueCacheFD != INVALID_FILE) {
        MNNCloseFile(mValueCacheFD);
        mValueCacheFD = INVALID_FILE;
        if (MNNRemoveFile(pathv.c_str()) != MNN::NO_ERROR) {
            MNN_PRINT("Failed to remove the file: %s\n", pathv.c_str());
        }
    }
}

void KVCacheManager::resetKVCacheFileSize(size_t keySize, size_t valueSize) {
    if (MNNSetFileSize(mKeyCacheFD, keySize) != MNN::NO_ERROR || MNNSetFileSize(mValueCacheFD, valueSize) != MNN::NO_ERROR) {
        MNN_PRINT("Failed to resize the kvcache files!\n");
    }
}

/*
**  @brief  Memory-map the kvcache file
**  @hint   After memory-mapping, we can access the kvcache files with pointers, just like accessing memory buffer
**          But the data actually resides in disk.
**          The OS will set some kernel page cache and manage the data swaping, which we do not need to care.
*/
void KVCacheManager::mmapKVCache(size_t keySize, size_t valueSize, file_t specKeyFile, file_t specValueFile)
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
