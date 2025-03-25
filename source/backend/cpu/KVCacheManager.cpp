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

void KVCacheManager::createKVCacheFile() {
    // Each layer has its own kvcache, so we have to create a key file and a value file for each layer and the file name must be unique
    // Here we use the address of the mResource as the file name because the addresses of mResource in different layers are guaranteed to be different
    std::string fileName = addrToHex(this);
    std::string pathk    = MNNFilePathConcat(mConfig.mKVCacheDir, fileName) + ".k";
    std::string pathv    = MNNFilePathConcat(mConfig.mKVCacheDir, fileName) + ".v";
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
    std::string fileName = addrToHex(this);
    std::string pathk = MNNFilePathConcat(mConfig.mKVCacheDir, fileName) + ".k";
    std::string pathv = MNNFilePathConcat(mConfig.mKVCacheDir, fileName) + ".v";
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
void KVCacheManager::mmapKVCache(size_t keySize, size_t valueSize)
{
    if (mMapKeyAddr == nullptr) {
        mMapKeyAddr = (char *)MNNMmapFile(mKeyCacheFD, keySize);
        if (mMapKeyAddr == nullptr) {
            MNN_PRINT("Failed to memory-map the kvcache!\n");
        }
    }
    if (mMapValueAddr == nullptr) {
        mMapValueAddr = (char *)MNNMmapFile(mValueCacheFD, valueSize);
        if (mMapValueAddr == nullptr) {
            MNN_PRINT("Failed to memory-map the kvcache!\n");
        }
    }
}

void KVCacheManager::unmapKVCache(size_t keySize, size_t valueSize)
{
    if (mMapKeyAddr != nullptr) {
        MNNUnmapFile(mMapKeyAddr, keySize);
        mMapKeyAddr = nullptr;
    }
    if (mMapValueAddr != nullptr) {
        MNNUnmapFile(mMapValueAddr, valueSize);
        mMapValueAddr = nullptr;
    }
}

/*
**  @brief  Expand the size of kvcache and copy it from the old tensor in memory to the new tensor in memory
**          Finally reset the pointer to the new tensor
*/
void KVCacheManager::expandKVCacheInMem(int oldMaxLength) {
    /*===================================  Key  ===================================*/
    if (mConfig.mUseInt8Kernel) {
        auto new_key = Tensor::createDevice<int8_t>({mKvNumHead, UP_DIV(mMaxLength, hP8), UP_DIV(mHeadDim, lP8), hP8 * lP8});
        mBackend->onAcquireBuffer(new_key, Backend::STATIC);
        for (int h = 0; h < mKvNumHead; h++) {
            memcpy(
                new_key->host<char>() + h * UP_DIV(mMaxLength, hP8) * UP_DIV(mHeadDim, lP8) * hP8 * lP8,
                mPastKey->host<char>() + h * UP_DIV(oldMaxLength, hP8) * UP_DIV(mHeadDim, lP8) * hP8 * lP8,
                UP_DIV(oldMaxLength, hP8) * UP_DIV(mHeadDim, lP8) * hP8 * lP8
            );
        }
        mPastKey.reset(new_key);
    }
    else if (mConfig.mQuantKey) {
        auto new_key = Tensor::createDevice<int8_t>({mKvNumHead, UP_DIV(mMaxLength, hP), mHeadDim, hP});
        mBackend->onAcquireBuffer(new_key, Backend::STATIC);
        for (int h = 0; h < mKvNumHead; h++) {
            memcpy(
                new_key->host<char>() + h * UP_DIV(mMaxLength, hP) * mHeadDim * hP,
                mPastKey->host<char>() + h * UP_DIV(oldMaxLength, hP) * mHeadDim * hP,
                UP_DIV(oldMaxLength, hP) * mHeadDim * hP
            );
        }
        mPastKey.reset(new_key);
    }
    else {
        auto new_key = Tensor::createDevice<float>({mKvNumHead, UP_DIV(mMaxLength, hP), mHeadDim, hP});
        mBackend->onAcquireBuffer(new_key, Backend::STATIC);
        for (int h = 0; h < mKvNumHead; h++) {
            memcpy(
                new_key->host<char>() + h * UP_DIV(mMaxLength, hP) * mHeadDim * hP * mBytes,
                mPastKey->host<char>() + h * UP_DIV(oldMaxLength, hP) * mHeadDim * hP * mBytes,
                UP_DIV(oldMaxLength, hP) * mHeadDim * hP * mBytes
            );
        }
        mPastKey.reset(new_key);
    }
    /*===================================  Value  ===================================*/
    if (mConfig.mQuantValue) {
        auto new_value = Tensor::createDevice<fp8_t>({mKvNumHead, UP_DIV(mHeadDim, hP), mMaxLength, hP});
        mBackend->onAcquireBuffer(new_value, Backend::STATIC);
        for (int h = 0; h < mKvNumHead; h++) {
            for (int i = 0; i < UP_DIV(mHeadDim, hP); i++) {
                memcpy(
                    new_value->host<char>() + (h * UP_DIV(mHeadDim, hP) + i) * mMaxLength * hP,
                    mPastValue->host<char>() + (h * UP_DIV(mHeadDim, hP) + i) * oldMaxLength * hP,
                    oldMaxLength * hP
                );
            }
        }
        mPastValue.reset(new_value);
    }
    else {
        auto new_value = Tensor::createDevice<float>({mKvNumHead, UP_DIV(mHeadDim, hP), mMaxLength, hP});
        mBackend->onAcquireBuffer(new_value, Backend::STATIC);
        for (int h = 0; h < mKvNumHead; h++) {
            for (int i = 0; i < UP_DIV(mHeadDim, hP); i++) {
                memcpy(
                    new_value->host<char>() + (h * UP_DIV(mHeadDim, hP) + i) * mMaxLength * hP * mBytes,
                    mPastValue->host<char>() + (h * UP_DIV(mHeadDim, hP) + i) * oldMaxLength * hP * mBytes,
                    oldMaxLength * hP * mBytes
                );
            }
        }
        mPastValue.reset(new_value);
    }
}

/*
**  @brief  Move the kvcache from memory to the memory-mapped kvcache files in disk
**          Then release the memory buffer of old kvcache
*/
void KVCacheManager::moveKVCacheFromMemToDisk(int oldMaxLength) {
    /*===================================  Key  ===================================*/
    if (mConfig.mUseInt8Kernel) {
        for (int h = 0; h < mKvNumHead; h++) {
            memcpy(
                mMapKeyAddr + h * UP_DIV(mMaxLength, hP8) * UP_DIV(mHeadDim, lP8) * hP8 * lP8,
                mPastKey->host<char>() + h * UP_DIV(oldMaxLength, hP8) * UP_DIV(mHeadDim, lP8) * hP8 * lP8,
                UP_DIV(oldMaxLength, hP8) * UP_DIV(mHeadDim, lP8) * hP8 * lP8
            );
        }
        mBackend->onReleaseBuffer(mPastKey.get(), Backend::STATIC);
        mPastKey.reset();
    }
    if (mConfig.mQuantKey) {
        for (int h = 0; h < mKvNumHead; h++) {
            memcpy(
                mMapKeyAddr + h * UP_DIV(mMaxLength, hP) * mHeadDim * hP,
                mPastKey->host<char>() + h * UP_DIV(oldMaxLength, hP) * mHeadDim * hP,
                UP_DIV(oldMaxLength, hP) * mHeadDim * hP
            );
        }
        mBackend->onReleaseBuffer(mPastKey.get(), Backend::STATIC);
        mPastKey.reset();
    }
    else {
        for (int h = 0; h < mKvNumHead; h++) {
            memcpy(
                mMapKeyAddr + h * UP_DIV(mMaxLength, hP) * mHeadDim * hP * mBytes,
                mPastKey->host<char>() + h * UP_DIV(oldMaxLength, hP) * mHeadDim * hP * mBytes,
                UP_DIV(oldMaxLength, hP) * mHeadDim * hP * mBytes
            );
        }
        mBackend->onReleaseBuffer(mPastKey.get(), Backend::STATIC);
        mPastKey.reset();
    }
    /*===================================  Value  ===================================*/
    if (mConfig.mQuantValue) {
        for (int h = 0; h < mKvNumHead; h++) {
            for (int i = 0; i < UP_DIV(mHeadDim, hP); i++) {
                memcpy(
                    mMapValueAddr + (h * UP_DIV(mHeadDim, hP) + i) * mMaxLength * hP,
                    mPastValue->host<char>() + (h * UP_DIV(mHeadDim, hP) + i) * oldMaxLength * hP,
                    oldMaxLength * hP
                );
            }
        }
        mBackend->onReleaseBuffer(mPastValue.get(), Backend::STATIC);
        mPastValue.reset();
    }
    else {
        for (int h = 0; h < mKvNumHead; h++) {
            for (int i = 0; i < UP_DIV(mHeadDim, hP); i++) {
                memcpy(
                    mMapValueAddr + (h * UP_DIV(mHeadDim, hP) + i) * mMaxLength * hP * mBytes,
                    mPastValue->host<char>() + (h * UP_DIV(mHeadDim, hP) + i) * oldMaxLength * hP * mBytes,
                    oldMaxLength * hP * mBytes
                );
            }
        }
        mBackend->onReleaseBuffer(mPastValue.get(), Backend::STATIC);
        mPastValue.reset();
    }
}

/*
**  @brief  Expand the size of kvcache files in disk
*/
void KVCacheManager::expandKVCacheInDisk(int oldMaxLength, int oldKeySize, int oldValueSize, int keySize, int valueSize) {
    // Step 1: Copy the old kvcache from files to temporary buffers in memory
    std::shared_ptr<Tensor> old_key, old_value;
    if (mConfig.mUseInt8Kernel) {
        old_key.reset(Tensor::createDevice<int8_t>({mKvNumHead, UP_DIV(oldMaxLength, hP8), UP_DIV(mHeadDim, lP8), hP8 * lP8}));
    } else if (mConfig.mQuantKey) {
        old_key.reset(Tensor::createDevice<int8_t>({mKvNumHead, UP_DIV(oldMaxLength, hP), mHeadDim, hP}));
    } else {
        old_key.reset(Tensor::createDevice<float>({mKvNumHead, UP_DIV(oldMaxLength, hP), mHeadDim, hP}));  
    }
    if (mConfig.mQuantValue) {
        old_value.reset(Tensor::createDevice<fp8_t>({mKvNumHead, UP_DIV(mHeadDim, hP), oldMaxLength, hP}));
    } else {
        old_value.reset(Tensor::createDevice<float>({mKvNumHead, UP_DIV(mHeadDim, hP), oldMaxLength, hP}));
    }
    mBackend->onAcquireBuffer(old_key.get(), Backend::STATIC);
    mBackend->onAcquireBuffer(old_value.get(), Backend::STATIC);
    mmapKVCache(oldKeySize, oldValueSize);
    memcpy(old_key->host<char>(),   mMapKeyAddr,   oldKeySize);
    memcpy(old_value->host<char>(), mMapValueAddr, oldValueSize);
    // Step 2: Resize the kvcache files and remap them
    unmapKVCache(oldKeySize, oldValueSize);
    resetKVCacheFileSize(keySize, valueSize);
    mmapKVCache(keySize, valueSize);
    // Step 3: Move the kvcache from temporary buffers in memory to disk
    if (mConfig.mUseInt8Kernel) {
        for (int h = 0; h < mKvNumHead; h++) {
            memcpy(
                mMapKeyAddr + h * UP_DIV(mMaxLength, hP8) * UP_DIV(mHeadDim, lP8) * hP8 * lP8,
                old_key->host<char>() + h * UP_DIV(oldMaxLength, hP8) * UP_DIV(mHeadDim, lP8) * hP8 * lP8,
                UP_DIV(oldMaxLength, hP8) * UP_DIV(mHeadDim, lP8) * hP8 * lP8
            );
        }
    } else if (mConfig.mQuantKey) {
        for (int h = 0; h < mKvNumHead; h++) {
            memcpy(
                mMapKeyAddr + h * UP_DIV(mMaxLength, hP) * mHeadDim * hP,
                old_key->host<char>() + h * UP_DIV(oldMaxLength, hP) * mHeadDim * hP,
                UP_DIV(oldMaxLength, hP) * mHeadDim * hP
            );
        }
    } else {
        for (int h = 0; h < mKvNumHead; h++) {
            memcpy(
                mMapKeyAddr + h * UP_DIV(mMaxLength, hP) * mHeadDim * hP * mBytes,
                old_key->host<char>() + h * UP_DIV(oldMaxLength, hP) * mHeadDim * hP * mBytes,
                UP_DIV(oldMaxLength, hP) * mHeadDim * hP * mBytes
            );
        }
    }
    if (mConfig.mQuantValue) {
        for (int h = 0; h < mKvNumHead; h++) {
            for (int i = 0; i < UP_DIV(mHeadDim, hP); i++) {
                memcpy(
                    mMapValueAddr + (h * UP_DIV(mHeadDim, hP) + i) * mMaxLength * hP,
                    old_value->host<char>() + (h * UP_DIV(mHeadDim, hP) + i) * oldMaxLength * hP,
                    oldMaxLength * hP
                );
            }
        }
    } else {
        for (int h = 0; h < mKvNumHead; h++) {
            for (int i = 0; i < UP_DIV(mHeadDim, hP); i++) {
                memcpy(
                    mMapValueAddr + (h * UP_DIV(mHeadDim, hP) + i) * mMaxLength * hP * mBytes,
                    old_value->host<char>() + (h * UP_DIV(mHeadDim, hP) + i) * oldMaxLength * hP * mBytes,
                    oldMaxLength * hP * mBytes
                );
            }
        }
    }
    // Step 4: Release the temporary buffers
    mBackend->onReleaseBuffer(old_key.get(), Backend::STATIC);
    mBackend->onReleaseBuffer(old_value.get(), Backend::STATIC);
}

void KVCacheManager::onResize(int kv_num_head, int head_dim) {
    mKvNumHead = kv_num_head;
    mHeadDim = head_dim;
    auto core  = static_cast<CPUBackend *>(mBackend)->functions();
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    mBytes = core->bytes;
    mThreadNum = static_cast<CPUBackend *>(mBackend)->threadNumber();
    if (mThreadNum > mKvNumHead) {
        mThreadNum = mKvNumHead;
    }
    if (mConfig.mUseInt8Kernel) {
        static_cast<CPUBackend *>(mBackend)->int8Functions()->MNNGetGemmUnit(&hP8, &lP8, &eP8);
    }
}

void KVCacheManager::onAlloc(int kv_seq_len) {
    mMaxLength = kv_seq_len + mConfig.mExpandChunk;
    size_t keySize = 0, valueSize = 0;
    if (mConfig.mUseInt8Kernel) {
        keySize = (size_t)mKvNumHead * UP_DIV(mMaxLength, hP8) * UP_DIV(mHeadDim, lP8) * hP8 * lP8;
    } else if (mConfig.mQuantKey) {
        keySize = (size_t)mKvNumHead * UP_DIV(mMaxLength, hP) * mHeadDim * hP;
    } else {
        keySize = (size_t)mKvNumHead * UP_DIV(mMaxLength, hP) * mHeadDim * hP * mBytes;
    }
    valueSize = (size_t)mKvNumHead * UP_DIV(mHeadDim, hP) * mMaxLength * hP * (mConfig.mQuantValue ? 1 : mBytes);
    /*============== Put the kvcache in disk ===========*/
    if (mConfig.mKVCacheSizeLimit != -1 && keySize + valueSize > mConfig.mKVCacheSizeLimit) {
        createKVCacheFile();
        resetKVCacheFileSize(keySize, valueSize);
        mmapKVCache(keySize, valueSize);
        mKVCacheInDisk = true;
    }
    /*============== Put the kvcache in memory ===========*/
    else {
        if (mConfig.mUseInt8Kernel) {
            mPastKey.reset(Tensor::createDevice<int8_t>({mKvNumHead, UP_DIV(mMaxLength, hP8), UP_DIV(mHeadDim, lP8), hP8 * lP8}));
        } else if (mConfig.mQuantKey) {
            mPastKey.reset(Tensor::createDevice<int8_t>({mKvNumHead, UP_DIV(mMaxLength, hP), mHeadDim, hP}));
        } else {
            mPastKey.reset(Tensor::createDevice<float>({mKvNumHead, UP_DIV(mMaxLength, hP), mHeadDim, hP}));
        }
        if (mConfig.mQuantValue) {
            mPastValue.reset(Tensor::createDevice<fp8_t>({mKvNumHead, UP_DIV(mHeadDim, hP), mMaxLength, hP}));
        } else {
            mPastValue.reset(Tensor::createDevice<float>({mKvNumHead, UP_DIV(mHeadDim, hP), mMaxLength, hP}));
        }
        mBackend->onAcquireBuffer(mPastKey.get(), Backend::STATIC); 
        mBackend->onAcquireBuffer(mPastValue.get(), Backend::STATIC); 
    }
    // scale, zero point and sum of key for quantization
    if (mConfig.mUseInt8Kernel) {
        mKeyScale.reset(Tensor::createDevice<int32_t>({mKvNumHead, UP_DIV(mMaxLength, hP8), hP8}));
        mKeyZeroPoint.reset(Tensor::createDevice<int32_t>({mKvNumHead, UP_DIV(mMaxLength, hP8), hP8}));
        mKeySum.reset(Tensor::createDevice<int32_t>({mKvNumHead, UP_DIV(mMaxLength, hP8), hP8}));
        mBackend->onAcquireBuffer(mKeyScale.get(), Backend::STATIC);
        mBackend->onAcquireBuffer(mKeyZeroPoint.get(), Backend::STATIC);
        mBackend->onAcquireBuffer(mKeySum.get(), Backend::STATIC);
    } else if (mConfig.mQuantKey) {
        mKeyScale.reset(Tensor::createDevice<float>({mKvNumHead, UP_DIV(mMaxLength, hP), hP}));
        mKeyZeroPoint.reset(Tensor::createDevice<float>({mKvNumHead, UP_DIV(mMaxLength, hP), hP}));
        mBackend->onAcquireBuffer(mKeyScale.get(), Backend::STATIC);
        mBackend->onAcquireBuffer(mKeyZeroPoint.get(), Backend::STATIC);
    }
}

void KVCacheManager::onRealloc(const KVMeta* meta) {
    auto kv_seq_len = meta->previous + meta->add - meta->remove + meta->computeReverseSize();
    if (kv_seq_len > mMaxLength) {
        // Realloc
        int oldMaxLength = mMaxLength;
        mMaxLength = kv_seq_len + mConfig.mExpandChunk;
        size_t oldKeySize, oldValueSize, keySize, valueSize;
        if (mConfig.mUseInt8Kernel) {
            oldKeySize = (size_t)mKvNumHead * UP_DIV(oldMaxLength, hP8) * UP_DIV(mHeadDim, lP8) * hP8 * lP8;
            keySize = (size_t)mKvNumHead * UP_DIV(mMaxLength, hP8) * UP_DIV(mHeadDim, lP8) * hP8 * lP8;
        } else if (mConfig.mQuantKey) {
            oldKeySize = (size_t)mKvNumHead * UP_DIV(oldMaxLength, hP) * mHeadDim * hP;
            keySize = (size_t)mKvNumHead * UP_DIV(mMaxLength, hP) * mHeadDim * hP;
        } else {
            oldKeySize = (size_t)mKvNumHead * UP_DIV(oldMaxLength, hP) * mHeadDim * hP * mBytes;
            keySize = (size_t)mKvNumHead * UP_DIV(mMaxLength, hP) * mHeadDim * hP * mBytes;
        }
        oldValueSize = (size_t)mKvNumHead * UP_DIV(mHeadDim, hP) * oldMaxLength * hP * (mConfig.mQuantValue ? 1 : mBytes);
        valueSize = (size_t)mKvNumHead * UP_DIV(mHeadDim, hP) * mMaxLength * hP * (mConfig.mQuantValue ? 1 : mBytes);
        /*==== No limit for kvcache ====*/
        if (mConfig.mKVCacheSizeLimit == -1) {
            expandKVCacheInMem(oldMaxLength);
        }
        /*==== Last time the kvcache is memory, now it should be in memory too ====*/
        else if (keySize + valueSize <= mConfig.mKVCacheSizeLimit) {
            expandKVCacheInMem(oldMaxLength);
        }
        /*==== Last time the kvcache is in memory, but now it should be moved to disk ====*/
        else if (oldKeySize + oldValueSize <= mConfig.mKVCacheSizeLimit) {
            createKVCacheFile();
            resetKVCacheFileSize(keySize, valueSize);
            mmapKVCache(keySize, valueSize);
            moveKVCacheFromMemToDisk(oldMaxLength);
            mKVCacheInDisk = true;
        }
        /*==== Last time the kvcache is disk, now it should be in disk too ====*/
        else {
            expandKVCacheInDisk(oldMaxLength, oldKeySize, oldValueSize, keySize, valueSize);
        }
        /* No matter where is the kvcache, the scales and zero points are always in memory, since their size is very small */
        if (mConfig.mUseInt8Kernel) {
            auto new_scale = Tensor::createDevice<int32_t>({mKvNumHead, UP_DIV(mMaxLength, hP8), hP8});
            auto new_zeroPoint = Tensor::createDevice<int32_t>({mKvNumHead, UP_DIV(mMaxLength, hP8), hP8});
            auto new_sum = Tensor::createDevice<int32_t>({mKvNumHead, UP_DIV(mMaxLength, hP8), hP8});
            mBackend->onAcquireBuffer(new_scale, Backend::STATIC);
            mBackend->onAcquireBuffer(new_zeroPoint, Backend::STATIC);
            mBackend->onAcquireBuffer(new_sum, Backend::STATIC);
            for (int h = 0; h < mKvNumHead; h++) {
                memcpy(new_scale->host<char>() + h * UP_DIV(mMaxLength, hP8) * hP8 * 4, mKeyScale->host<char>() + h * UP_DIV(oldMaxLength, hP8) * hP8 * 4, UP_DIV(oldMaxLength, hP8) * hP8 * 4);
                memcpy(new_zeroPoint->host<char>() + h * UP_DIV(mMaxLength, hP8) * hP8 * 4, mKeyZeroPoint->host<char>() + h * UP_DIV(oldMaxLength, hP8) * hP8 * 4, UP_DIV(oldMaxLength, hP8) * hP8 * 4);
                memcpy(new_sum->host<char>() + h * UP_DIV(mMaxLength, hP8) * hP8 * 4, mKeySum->host<char>() + h * UP_DIV(oldMaxLength, hP8) * hP8 * 4, UP_DIV(oldMaxLength, hP8) * hP8 * 4);
            }
            mKeyScale.reset(new_scale);
            mKeyZeroPoint.reset(new_zeroPoint);
            mKeySum.reset(new_sum);
        } else if (mConfig.mQuantKey) {
            auto new_scale = Tensor::createDevice<float>({mKvNumHead, UP_DIV(mMaxLength, hP), 1, hP});
            auto new_zeroPoint = Tensor::createDevice<float>({mKvNumHead, UP_DIV(mMaxLength, hP), 1, hP});
            mBackend->onAcquireBuffer(new_scale, Backend::STATIC);
            mBackend->onAcquireBuffer(new_zeroPoint, Backend::STATIC);
            for (int h = 0; h < mKvNumHead; h++) {
                memcpy(new_scale->host<char>() + h * UP_DIV(mMaxLength, hP) * hP * mBytes, mKeyScale->host<char>() + h * UP_DIV(oldMaxLength, hP) * hP * mBytes, UP_DIV(oldMaxLength, hP) * hP * mBytes);
                memcpy(new_zeroPoint->host<char>() + h * UP_DIV(mMaxLength, hP) * hP * mBytes, mKeyZeroPoint->host<char>() + h * UP_DIV(oldMaxLength, hP) * hP * mBytes, UP_DIV(oldMaxLength, hP) * hP * mBytes);
            }
            mKeyScale.reset(new_scale);
            mKeyZeroPoint.reset(new_zeroPoint);
        }
    }
    // Remove
    auto start = mPastLength - meta->remove;
    if (0 == meta->n_reserve) {
        mPastLength = start;
        return;
    }
    // Don't support not align reserve
    auto align = hP;
    auto dstStart = start;
    auto lastValidSrcEnd = start;
    for (int n=0; n<meta->n_reserve; ++n) {
        auto lastEndAlign = UP_DIV(lastValidSrcEnd, align) * align;
        auto begin = meta->reserve[2 * n];
        auto size = meta->reserve[2 * n + 1];
        auto startAlign = ((begin + start) / align) * align;
        if (startAlign <= lastEndAlign) {
            // Fullly reserve
            dstStart = dstStart + size;
            lastValidSrcEnd = begin + start + size;
            continue;
        }
        auto end = begin + start + size;
        auto endAlign = UP_DIV(end, align) * align;
        
        auto sizeUnit = (endAlign - startAlign) / align;
        auto dstStartAlign = UP_DIV(dstStart, align) * align;

        //TODO: Support Quant
//        mPastKey.reset(Tensor::createDevice<float>({mKvNumHead, UP_DIV(mMaxLength, hP), mHeadDim, hP}));

        // Move K
        auto keyStride = UP_DIV(mMaxLength, align) * align * mHeadDim;
        auto dstKAddr = keyAddr() + dstStartAlign * mHeadDim * mBytes;
        auto srcKAddr = keyAddr() + startAlign * mHeadDim * mBytes;
        for (int i=0; i<mKvNumHead; ++i) {
            auto dst = dstKAddr + i * keyStride * mBytes;
            auto src = srcKAddr + i * keyStride * mBytes;
            for (int j=0; j<sizeUnit; ++j) {
                ::memcpy(dst + j * align * mHeadDim * mBytes, src + j * align * mHeadDim * mBytes, align * mHeadDim * mBytes);
            }
        }

        //        mPastValue.reset(Tensor::createDevice<float>({mKvNumHead, UP_DIV(mHeadDim, hP), mMaxLength, hP}));

        // Move V
        auto dstVAddr = valudAddr() + dstStartAlign * align * mBytes;
        auto srcVAddr = valudAddr() + startAlign * align * mBytes;
        auto number = mKvNumHead * UP_DIV(mHeadDim, align);
        for (int i=0; i<number; ++i) {
            auto dst = dstVAddr + i * mMaxLength * align * mBytes;
            auto src = srcVAddr + i * mMaxLength * align * mBytes;
            for (int j=0; j<sizeUnit; ++j) {
                ::memcpy(dst + j * align * align * mBytes, src + j * align * align * mBytes, align * align * mBytes);
            }
        }
        dstStart = dstStart + size;
        lastValidSrcEnd = begin + start + size;
    }
    mPastLength = dstStart;
}

void KVCacheManager::onClear() {
    if (mKVCacheInDisk) {
        size_t keySize = 0, valueSize = 0;
        if (mConfig.mUseInt8Kernel) {
            keySize = (size_t)mKvNumHead * UP_DIV(mMaxLength, hP8) * UP_DIV(mHeadDim, lP8) * hP8 * lP8;
        } else if (mConfig.mQuantKey) {
            keySize = (size_t)mKvNumHead * UP_DIV(mMaxLength, hP) * mHeadDim * hP;
        } else {
            keySize = (size_t)mKvNumHead * UP_DIV(mMaxLength, hP) * mHeadDim * hP * mBytes;
        }
        valueSize = (size_t)mKvNumHead * UP_DIV(mHeadDim, hP) * mMaxLength * hP * (mConfig.mQuantValue ? 1 : mBytes);    
        unmapKVCache(keySize, valueSize);
        removeKVCacheFile();
        mKVCacheInDisk = false;
    }
    mPastKey.reset();
    mPastValue.reset();
    mKeyScale.reset();
    mKeyZeroPoint.reset();
    mKeySum.reset();
    mMaxLength = mPastLength = 0;
}

template <typename T>
void KVCacheManager::pack_key(const Tensor* key, int seq_len, int kv_h) {
    if (mConfig.mUseInt8Kernel) {  // [maxlen/hP8, headdim/lP8, hP8, lP8]
        int8_t * key_dst = reinterpret_cast<int8_t*>(addrOfKey(kv_h));
        float * scale_dst = reinterpret_cast<float*>(addrOfScale(kv_h));
        float * zeroPoint_dst = reinterpret_cast<float*>(addrOfZeroPoint(kv_h));
        float * sum_dst = reinterpret_cast<float*>(addrOfKeySum(kv_h));
        for (int s = 0; s < seq_len; s++) {
            T * key_src = key->host<T>() + s * mKvNumHead * mHeadDim + kv_h * mHeadDim;
            float minKey = key_src[0];
            float maxKey = key_src[0];
            float sumKey = key_src[0];
            for (int d = 1; d < mHeadDim; d++) {
                minKey = ALIMIN(minKey, key_src[d]);
                maxKey = ALIMAX(maxKey, key_src[d]);
                sumKey += key_src[d];
            }
            int out_index = (mPastLength + s) / hP8;
            int in_index  = (mPastLength + s) % hP8;
            scale_dst[out_index * hP8 + in_index] = (maxKey - minKey) / 255.0f;
            zeroPoint_dst[out_index * hP8 + in_index] = -255.0f * minKey / (maxKey - minKey) - 128.0;
            sum_dst[out_index * hP8 + in_index] = sumKey;
            for (int d = 0; d < mHeadDim; d++) {
                int i = d / lP8;
                int j = d % lP8;
                key_dst[out_index * UP_DIV(mHeadDim, lP8) * hP8 * lP8 + i * hP8 * lP8 + in_index * lP8 + j] = roundf((key_src[d] - minKey) / (maxKey - minKey) * 255.0f - 128.0f);
            }
        }
    }
    else if (mConfig.mQuantKey) {  // [maxlen/hP, headdim, hP]
        int8_t * key_dst = reinterpret_cast<int8_t*>(addrOfKey(kv_h));
        T * scale_dst = reinterpret_cast<T*>(addrOfScale(kv_h));
        T * zeroPoint_dst = reinterpret_cast<T*>(addrOfZeroPoint(kv_h));
        for (int i = 0; i < seq_len; i++) {
            T * key_src = key->host<T>() + i * mKvNumHead * mHeadDim + kv_h * mHeadDim;
            int out_index = (mPastLength + i) / hP;
            int in_index  = (mPastLength + i) % hP;
            T minKey, maxKey;
            static_cast<CPUBackend*>(mBackend)->functions()->MNNCountMaxMinValue((float*)key_src, (float*)&minKey, (float*)&maxKey, mHeadDim);
            scale_dst[out_index * hP + in_index] = (maxKey - minKey) / 255.0f;
            zeroPoint_dst[out_index * hP + in_index] = 128.0f * (maxKey - minKey) / 255.0f + minKey;
            for (int j = 0; j < mHeadDim; j++) {
                key_dst[out_index * mHeadDim * hP + j * hP + in_index] = roundf((key_src[j] - minKey) / (maxKey - minKey) * 255 - 128);
            }
        }
    }
    else { // [maxlen/hP, headdim, hP]
        T * key_dst = reinterpret_cast<T*>(addrOfKey(kv_h));
        for (int i = 0; i < seq_len; i++) {
            T * key_src = key->host<T>() + i * mKvNumHead * mHeadDim + kv_h * mHeadDim;
            int out_index = (mPastLength + i) / hP;
            int in_index  = (mPastLength + i) % hP;
            for (int j = 0; j < mHeadDim; j++) {
                key_dst[out_index * mHeadDim * hP + j * hP + in_index] = key_src[j];
            }
        }
    }
}

template <typename T>
void KVCacheManager::pack_value(const Tensor* value, int seq_len, int kv_h) { // [headdim/hP, maxlen, hP]
    if (mConfig.mQuantValue) {
        fp8_t * value_dst = reinterpret_cast<fp8_t*>(addrOfValue(kv_h));
        uint8_t * buf = (uint8_t *)MNNMemoryAllocAlign(mHeadDim, MNN_MEMORY_ALIGN_DEFAULT);
        for (int i = 0; i < seq_len; i++) {
            T * value_src = value->host<T>() + i * mKvNumHead * mHeadDim + kv_h * mHeadDim;
            if (sizeof(T) == 2) {
                static_cast<CPUBackend*>(mBackend)->functions()->MNNFp16ToFp8(buf, (uint16_t*)value_src, mHeadDim);
            } else {
                static_cast<CPUBackend*>(mBackend)->functions()->MNNFp32ToFp8(buf, (float*)value_src, mHeadDim);
            }
            for (int j = 0; j < mHeadDim; j++) {
                int out_index = j / hP;
                int in_index  = j % hP;
                value_dst[out_index * mMaxLength * hP + (mPastLength + i) * hP + in_index] = buf[j];
            }
        }
        MNNMemoryFreeAlign(buf);
    }
    else {
        T * value_dst = reinterpret_cast<T*>(addrOfValue(kv_h));
        for (int i = 0; i < seq_len; i++) {
            T * value_src = value->host<T>() + i * mKvNumHead * mHeadDim + kv_h * mHeadDim;
            for (int j = 0; j < mHeadDim; j++) {
                int out_index = j / hP;
                int in_index  = j % hP;
                value_dst[out_index * mMaxLength * hP + (mPastLength + i) * hP + in_index] = value_src[j];
            }
        }
    }
}

void KVCacheManager::onPushBack(const Tensor * key, const Tensor * value) {
    auto core = static_cast<CPUBackend*>(mBackend)->functions();
    int seq_len = key->length(1);
    int tileCount = UP_DIV(mKvNumHead, mThreadNum);
    std::function<void(int)> packKV = [=](int tid) {
        for (int kv_h = tid * tileCount; kv_h < (tid+1) * tileCount && kv_h < mKvNumHead; kv_h++) {
            if (mBytes == 2) {
                pack_key<FLOAT16_T>(key, seq_len, kv_h);
                pack_value<FLOAT16_T>(value, seq_len, kv_h);
            } else {
                pack_key<float>(key, seq_len, kv_h);
                pack_value<float>(value, seq_len, kv_h);
            }
        }
    };
    MNN_CONCURRENCY_BEGIN(tid, mThreadNum) {
        packKV((int)tid);
    }
    MNN_CONCURRENCY_END();
    mPastLength += seq_len;
}

void KVCacheManager::onDequantValue(Tensor * dequantedValues) {
    auto core = static_cast<CPUBackend*>(mBackend)->functions();
    int tileCount = UP_DIV(mKvNumHead, mThreadNum);
    std::function<void(int)> dequant = [=](int tid) {
        for (int kv_h = tid * tileCount; kv_h < (tid+1) * tileCount && kv_h < mKvNumHead; kv_h++) {
            char * dst = dequantedValues->host<char>() + kv_h * UP_DIV(mHeadDim, hP) * mPastLength * hP * mBytes;
            char * src = addrOfValue(kv_h);
            for (int i = 0; i < UP_DIV(mHeadDim, hP); i++) {
                if (mBytes == 2) {
                    core->MNNFp8ToFp16((uint16_t*)dst, (uint8_t*)src, mPastLength * hP);
                } else {
                    core->MNNFp8ToFp32((float*)dst, (uint8_t*)src, mPastLength * hP);
                }
                dst += mPastLength * hP * mBytes;
                src += mMaxLength * hP;
            }
        }
    };
    MNN_CONCURRENCY_BEGIN(tid, mThreadNum) {
        dequant((int)tid);
    }
    MNN_CONCURRENCY_END();
}

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
