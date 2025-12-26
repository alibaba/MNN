//
//  CPUKVCacheManager.cpp
//  MNN
//
//  Created by MNN on 2024/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "CPUKVCacheManager.hpp"
#include "core/Concurrency.h"

namespace MNN {

/*
**  @brief  Expand the size of kvcache and copy it from the old tensor in memory to the new tensor in memory
**          Finally reset the pointer to the new tensor
*/
void CPUKVCacheManager::expandKVCacheInMem(int oldMaxLength) {
    /*===================================  Key  ===================================*/
    auto new_key = Tensor::createDevice<int8_t>({mKvNumHead, (int)mCurrentKeySizePerHead});
    mBackend->onAcquireBuffer(new_key, Backend::STATIC);
    if (mQuantKey) {
        memset(new_key->host<int8_t>(), 0, mKvNumHead * mCurrentKeySizePerHead);
    }
    for (int h = 0; h < mKvNumHead; h++) {
        memcpy(
               new_key->host<int8_t>() + h * mCurrentKeySizePerHead,
               mPastKey->host<int8_t>() + h * mPastKey->stride(0),
               mPastKey->stride(0)
               );
        if (!mQuantKey && (new_key->stride(0) - mPastKey->stride(0)) > 0) {
            memset(new_key->host<int8_t>() + h * new_key->stride(0) + mPastKey->stride(0), 0, (new_key->stride(0) - mPastKey->stride(0)));
        }
    }
    mPastKey.reset(new_key);
    /*===================================  Value  ===================================*/
    auto newValue = Tensor::createDevice<int8_t>({mKvNumHead, (int)mCurrentValueSizePerHead});
    mBackend->onAcquireBuffer(newValue, Backend::STATIC);

    if (mUseFlashAttention) { // [mKvNumHead, UP_DIV(mMaxLength, mFlashAttentionUpperKv), UP_DIV(mHeadDim, hP), UP_DIV(mFlashAttentionUpperKv, lP), hP, lP]
        for (int h = 0; h < mKvNumHead; h++) {
            memset(newValue->host<int8_t>() + h * newValue->stride(0), 0, newValue->stride(0));
            memcpy(
                newValue->host<int8_t>() + h * newValue->stride(0),
                mPastValue->host<int8_t>() + h * mPastValue->stride(0),
                mPastValue->stride(0)
            );
        }
    } else {
        if (mQuantValue) { // [mKvNumHead, UP_DIV(mHeadDim, hP8), (UP_DIV(mMaxLength, lP8)*hP8*lP8+2*hP8*sizeof(float)) ]
            auto currentWeightInside = ROUND_UP(mMaxLength, lP8) * hP8;
            auto currentStride1 = currentWeightInside + 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES;
            auto currentStride0 = currentStride1 * UP_DIV(mHeadDim, hP8);

            auto prevWeightInside = ROUND_UP(oldMaxLength, lP8) * hP8;
            auto prevStride1 = prevWeightInside + 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES;
            auto prevStride0 = prevStride1 * UP_DIV(mHeadDim, hP8);
            for (int h = 0; h < mKvNumHead; ++h) {
                for (int d = 0; d < UP_DIV(mHeadDim, hP8); ++d) {
                    auto dstPtr = newValue->host<int8_t>() + h * currentStride0 + d * currentStride1;
                    auto srcPtr = mPastValue->host<int8_t>() + h * prevStride0 + d * prevStride1;

                    // initialize 0 for weightInt8
                    memset(dstPtr, 0, currentWeightInside);
                    // copy inner side weightInt8
                    memcpy(dstPtr, srcPtr, prevWeightInside);
                    // copy hP8 scale&bias
                    memcpy(dstPtr + currentWeightInside, srcPtr + prevWeightInside, 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES);
                }
            }
        } else { // [mKvNumHead, UP_DIV(mHeadDim, hP), UP_DIV(mMaxLength, lP), hP, lP]
            auto currentStride1 = ROUND_UP(mMaxLength, lP) * hP * mBytes;
            auto currentStride0 = ROUND_UP(mMaxLength, lP) * hP * UP_DIV(mHeadDim, hP) * mBytes;

            auto prevStride1 = ROUND_UP(oldMaxLength, lP) * hP * mBytes;
            auto prevStride0 = ROUND_UP(oldMaxLength, lP) * hP * UP_DIV(mHeadDim, hP) * mBytes;
            for (int h = 0; h < mKvNumHead; ++h) {
                for (int d = 0; d < UP_DIV(mHeadDim, hP); ++d) {
                    auto dstPtr = newValue->host<int8_t>() + h * currentStride0 + d * currentStride1;
                    auto srcPtr = mPastValue->host<int8_t>() + h * prevStride0 + d * prevStride1;

                    // initialize 0 for weight
                    if (lP > 1) {
                        memset(dstPtr, 0, currentStride1);
                    }
                    // copy inner side weight
                    memcpy(dstPtr, srcPtr, prevStride1);
                }
            }
        }
    }
    mPastValue.reset(newValue);
}

/*
**  @brief  Move the kvcache from memory to the memory-mapped kvcache files in disk
**          Then release the memory buffer of old kvcache
*/
void CPUKVCacheManager::moveKVCacheFromMemToDisk(int oldMaxLength) {
    /*===================================  Key  ===================================*/
    size_t prevKeySizePerHead = 0;
    if (mQuantKey) {
        prevKeySizePerHead = ROUND_UP(oldMaxLength, hP8) * ROUND_UP(mHeadDim, lP8) + 2 * QUANT_INFO_BYTES * mConfig.mBlockNum * ROUND_UP(oldMaxLength, hP8);
    } else  {
        prevKeySizePerHead = UP_DIV(oldMaxLength, hP) * ROUND_UP(mHeadDim, lP) * hP * mBytes;
    }
    if (mHeadDim % lP || mQuantKey) {
        memset(mMapKeyAddr, 0, mKvNumHead * mCurrentKeySizePerHead);
    }
    for (int h = 0; h < mKvNumHead; h++) {
        memcpy(
            mMapKeyAddr + h * mCurrentKeySizePerHead,
            mPastKey->host<int8_t>() + h * prevKeySizePerHead,
            prevKeySizePerHead
        );
    }
    mBackend->onReleaseBuffer(mPastKey.get(), Backend::STATIC);
    mPastKey.reset();
    /*===================================  Value  ===================================*/
    {
        size_t prevValueSizePerHead = 0;
        if (mQuantValue) {
            prevValueSizePerHead = UP_DIV(oldMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP8) * ROUND_UP(mFlashAttentionUpperKv, lP8) + 2 * QUANT_INFO_BYTES * mConfig.mBlockNum * ROUND_UP(mHeadDim, hP8));
        } else {
            prevValueSizePerHead = UP_DIV(oldMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP) * ROUND_UP(mFlashAttentionUpperKv, lP) * mBytes);
        }
        if (lP > 1 || mQuantValue) {
            memset(mMapValueAddr, 0, mKvNumHead * mCurrentValueSizePerHead);
        }

        if (mUseFlashAttention) {
            for (int h = 0; h < mKvNumHead; h++) {
                memcpy(
                       mMapValueAddr + h * mCurrentValueSizePerHead,
                       mPastValue->host<int8_t>() + h * prevValueSizePerHead,
                       prevValueSizePerHead
                       );
            }
        } else {
            if (mQuantValue) { // [mKvNumHead, UP_DIV(mHeadDim, hP8), (UP_DIV(mMaxLength, lP8)*hP8*lP8+2*hP8*sizeof(float)) ]
                auto currentWeightInside = ROUND_UP(mMaxLength, lP8) * hP8;
                auto currentStride1 = currentWeightInside + 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES;
                auto currentStride0 = currentStride1 * UP_DIV(mHeadDim, hP8);

                auto prevWeightInside = ROUND_UP(oldMaxLength, lP8) * hP8;
                auto prevStride1 = prevWeightInside + 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES;
                auto prevStride0 = prevStride1 * UP_DIV(mHeadDim, hP8);
                for (int h = 0; h < mKvNumHead; ++h) {
                    for (int d = 0; d < UP_DIV(mHeadDim, hP8); ++d) {
                        auto dstPtr = mMapValueAddr + h * currentStride0 + d * currentStride1;
                        auto srcPtr = mPastValue->host<int8_t>() + h * prevStride0 + d * prevStride1;

                        // initialize 0 for weightInt8
                        memset(dstPtr, 0, currentWeightInside);
                        // copy inner side weightInt8
                        memcpy(dstPtr, srcPtr, prevWeightInside);
                        // copy hP8 scale&bias
                        memcpy(dstPtr + currentWeightInside, srcPtr + prevWeightInside, 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES);
                    }
                }
            } else { // [mKvNumHead, UP_DIV(mHeadDim, hP), UP_DIV(mMaxLength, lP), hP, lP]
                auto currentStride1 = ROUND_UP(mMaxLength, lP) * hP * mBytes;
                auto currentStride0 = ROUND_UP(mMaxLength, lP) * hP * UP_DIV(mHeadDim, hP) * mBytes;

                auto prevStride1 = ROUND_UP(oldMaxLength, lP) * hP * mBytes;
                auto prevStride0 = ROUND_UP(oldMaxLength, lP) * hP * UP_DIV(mHeadDim, hP) * mBytes;
                for (int h = 0; h < mKvNumHead; ++h) {
                    for (int d = 0; d < UP_DIV(mHeadDim, hP); ++d) {
                        auto dstPtr = mMapValueAddr + h * currentStride0 + d * currentStride1;
                        auto srcPtr = mPastValue->host<int8_t>() + h * prevStride0 + d * prevStride1;

                        // initialize 0 for weight
                        if (lP > 1) {
                            memset(dstPtr, 0, currentStride1);
                        }
                        // copy inner side weight
                        memcpy(dstPtr, srcPtr, prevStride1);
                    }
                }
            }
        }
        mBackend->onReleaseBuffer(mPastValue.get(), Backend::STATIC);
        mPastValue.reset();
    }
}

/*
**  @brief  Expand the size of kvcache files in disk
*/
void CPUKVCacheManager::expandKVCacheInDisk(int oldMaxLength, int oldKeySize, int oldValueSize, int keySize, int valueSize, file_t specKeyFile, file_t specValueFile) {
    // Step 1: Copy the old kvcache from files to temporary buffers in memory
    auto prevKeySizePerHead = oldKeySize / mKvNumHead;
    auto prevValueSizePerHead = oldValueSize / mKvNumHead;
    std::shared_ptr<Tensor> prevKey, prevValue;
    prevKey.reset(Tensor::createDevice<int8_t>({mKvNumHead, prevKeySizePerHead}));
    prevValue.reset(Tensor::createDevice<int8_t>({mKvNumHead, prevValueSizePerHead}));

    mBackend->onAcquireBuffer(prevKey.get(), Backend::STATIC);
    mBackend->onAcquireBuffer(prevValue.get(), Backend::STATIC);
    if (mHeadDim % lP) {
        memset(prevKey->host<uint8_t>(), 0, prevKey->length(0) * prevKey->stride(0));
    }
    if (lP > 1) {
        // can't be mMaxLenth % lP, since mMaxLength may be larger than seq_len for prefilling, we should ensure the (mMaxLength - seq_len)'s buffer is 0.
        // computing L is seq_len
        memset(prevValue->host<uint8_t>(), 0, prevValue->length(0) * prevValue->stride(0));
    }
    mmapKVCache(oldKeySize, oldValueSize, specKeyFile, specValueFile);
    memcpy(prevKey->host<int8_t>(),   mMapKeyAddr,   oldKeySize);
    memcpy(prevValue->host<int8_t>(), mMapValueAddr, oldValueSize);
    // Step 2: Resize the kvcache files and remap them
    unmapKVCache(oldKeySize, oldValueSize);
    resetKVCacheFileSize(keySize, valueSize);
    mmapKVCache(keySize, valueSize);
    // Step 3: Move the kvcache from temporary buffers in memory to disk
    memset(mMapKeyAddr, 0, keySize);
    memset(mMapValueAddr, 0, valueSize);

    for (int h = 0; h < mKvNumHead; h++) {
        memcpy(mMapKeyAddr + h * mCurrentKeySizePerHead, prevKey->host<int8_t>() + h * prevKeySizePerHead, prevKeySizePerHead);
    }

    if (mUseFlashAttention) {
        for (int h = 0; h < mKvNumHead; h++) {
            memcpy(mMapValueAddr + h * mCurrentValueSizePerHead, prevValue->host<int8_t>() + h * prevValueSizePerHead, prevValueSizePerHead);
        }
    } else {
        if (mQuantValue) {
            auto currentWeightInside = ROUND_UP(mMaxLength, lP8) * hP8;
            auto currentStride1 = currentWeightInside + 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES;
            auto currentStride0 = currentStride1 * UP_DIV(mHeadDim, hP8);

            auto prevWeightInside = ROUND_UP(oldMaxLength, lP8) * hP8;
            auto prevStride1 = prevWeightInside + 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES;
            auto prevStride0 = prevStride1 * UP_DIV(mHeadDim, hP8);

            for (int h = 0; h < mKvNumHead; ++h) {
                for (int d = 0; d < UP_DIV(mHeadDim, hP8); ++d) {
                    auto dstPtr = mMapValueAddr + h * currentStride0 + d * currentStride1;
                    auto srcPtr = prevValue->host<int8_t>() + h * prevStride0 + d * prevStride1;

                    // initialize 0 for weightInt8
                    memset(dstPtr, 0, currentWeightInside);
                    // copy inner side weightInt8
                    memcpy(dstPtr, srcPtr, prevWeightInside);
                    // copy hP8 scale&bias
                    memcpy(dstPtr + currentWeightInside, srcPtr + prevWeightInside, 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES);
                }
            }
        } else {
            auto currentStride1 = ROUND_UP(mMaxLength, lP) * hP * mBytes;
            auto currentStride0 = ROUND_UP(mMaxLength, lP) * hP * UP_DIV(mHeadDim, hP) * mBytes;

            auto prevStride1 = ROUND_UP(oldMaxLength, lP) * hP * mBytes;
            auto prevStride0 = ROUND_UP(oldMaxLength, lP) * hP * UP_DIV(mHeadDim, hP) * mBytes;
            for (int h = 0; h < mKvNumHead; ++h) {
                for (int d = 0; d < UP_DIV(mHeadDim, hP); ++d) {
                    auto dstPtr = mMapValueAddr + h * currentStride0 + d * currentStride1;
                    auto srcPtr = prevValue->host<int8_t>() + h * prevStride0 + d * prevStride1;

                    // initialize 0 for weight
                    if (lP > 1) {
                        memset(dstPtr, 0, currentStride1);
                    }
                    // copy inner side weight
                    memcpy(dstPtr, srcPtr, prevStride1);
                }
            }
        }
    }

    // Step 4: Release the temporary buffers
    mBackend->onReleaseBuffer(prevKey.get(), Backend::STATIC);
    mBackend->onReleaseBuffer(prevValue.get(), Backend::STATIC);
}

void CPUKVCacheManager::onResize(int kv_num_head, int head_dim) {
    mKvNumHead = kv_num_head;
    mHeadDim = head_dim;
    auto core  = static_cast<CPUBackend *>(mBackend)->functions();
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    mBytes = core->bytes;
    mThreadNum = static_cast<CPUBackend *>(mBackend)->threadNumber();
    if (mThreadNum > mKvNumHead) {
        mThreadNum = mKvNumHead;
    }

    static_cast<CPUBackend *>(mBackend)->int8Functions()->MNNGetGemmUnit(&hP8, &lP8, &eP8);
    mQuantKeyFunc = core->MNNQuantAttentionKey;
    mQuantValueFunc = core->MNNQuantAttentionValue;

}

void CPUKVCacheManager::onAlloc(KVMeta* meta, int seq_len) {
    mMeta = meta;

    // load disk prefix kvcache
    if(mMeta != nullptr && mMeta->file_name.size() > 0 && mMeta->file_flag == KVMeta::PendingRead) {
        // create new files
        std::string pathk    = MNNFilePathConcat(mConfig.mPrefixCacheDir, mMeta->file_name) + "_" + std::to_string(mMeta->layer_index) + ".k";
        std::string pathv    = MNNFilePathConcat(mConfig.mPrefixCacheDir, mMeta->file_name) + "_" + std::to_string(mMeta->layer_index++) + ".v";
        mMeta->layer_index = mMeta->layer_index % mMeta->layer_nums;
        auto old_key_fd   = MNNOpenFile(pathk.c_str(), MNN_FILE_WRITE);
        auto old_value_fd = MNNOpenFile(pathv.c_str(), MNN_FILE_WRITE);
        if (old_key_fd == INVALID_FILE) {
            MNN_PRINT("Failed to open the file: %s\n", pathk.c_str());
        }
        if (old_value_fd == INVALID_FILE) {
            MNN_PRINT("Failed to open the file: %s\n", pathv.c_str());
        }

        // get kv cache file info
        auto oldKeySize = MNNGetFileSize(old_key_fd);
        auto oldValueSize = MNNGetFileSize(old_value_fd);

        size_t oldMaxLength = 0;
        if (mQuantKey || mQuantValue) {
            MNN_ERROR("[Error]: Currently, kvcache save in disk not support quantized key/value\n");
        } else {
            size_t oldKeyMaxLength = oldKeySize / (mKvNumHead * ROUND_UP(mHeadDim, lP) * mBytes);
            size_t oldValueMaxLength = oldValueSize / (mKvNumHead * ROUND_UP(mHeadDim, hP) * mBytes);
            oldMaxLength = ALIMIN(oldKeyMaxLength, oldValueMaxLength);
        }
        if(oldMaxLength < meta->seqlen_in_disk) {
            MNN_ERROR("[Error]: Kvcache in disk size smaller than saved lengthInDiskToload:%d\n", (int)meta->seqlen_in_disk);
        }

        if (mUseFlashAttention) {
            setFlashAttentionUpperKv(MNN_FLASH_ATTENTION_BLOCK_SIZE);
        } else {
            setFlashAttentionUpperKv(mMaxLength);
        }
        int kv_seq_len = meta->add + meta->seqlen_in_disk;
        mMaxLength = kv_seq_len > oldMaxLength ? kv_seq_len + mConfig.mExpandChunk : oldMaxLength;
        size_t keySize = (size_t)mKvNumHead * ROUND_UP(mMaxLength, hP) * ROUND_UP(mHeadDim, lP) * mBytes;
        size_t valueSize = (size_t)mKvNumHead * UP_DIV(mMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP) * ROUND_UP(mFlashAttentionUpperKv, lP) * mBytes);

        keySize = ALIMAX(keySize, oldKeySize);
        valueSize = ALIMAX(valueSize, oldValueSize);

        if (mQuantKey) {
            mCurrentKeySizePerHead = ROUND_UP(mMaxLength, hP8) * ROUND_UP(mHeadDim, lP8) + 2 * QUANT_INFO_BYTES * mConfig.mBlockNum * ROUND_UP(mMaxLength, hP8);
        } else {
            mCurrentKeySizePerHead = ROUND_UP(mMaxLength, hP) * ROUND_UP(mHeadDim, lP) * mBytes;
        }
        if (mQuantValue) {
            mCurrentValueSizePerHead = UP_DIV(mMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP8) * ROUND_UP(mFlashAttentionUpperKv, lP8) + 2 * QUANT_INFO_BYTES * mConfig.mBlockNum * ROUND_UP(mHeadDim, hP8));
        } else {
            mCurrentValueSizePerHead = UP_DIV(mMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP) * ROUND_UP(mFlashAttentionUpperKv, lP) * mBytes);
        }

        createKVCacheFile();
        resetKVCacheFileSize(keySize, valueSize);
        expandKVCacheInDisk(oldMaxLength, oldKeySize, oldValueSize, keySize, valueSize, old_key_fd, old_value_fd);
        mPastLength = meta->seqlen_in_disk;
        mKVCacheInDisk = true;

        return;
    }

    int kv_seq_len = mMeta != nullptr ? (int)meta->add : seq_len;
    mMaxLength = kv_seq_len + mConfig.mExpandChunk;
    if (mUseFlashAttention) {
        setFlashAttentionUpperKv(MNN_FLASH_ATTENTION_BLOCK_SIZE);
    } else {
        setFlashAttentionUpperKv(mMaxLength);
    }

    // 1. compute size
    if (mQuantKey) {
        mCurrentKeySizePerHead = ROUND_UP(mMaxLength, hP8) * ROUND_UP(mHeadDim, lP8) + 2 * QUANT_INFO_BYTES * mConfig.mBlockNum * ROUND_UP(mMaxLength, hP8);
    } else {
        mCurrentKeySizePerHead = ROUND_UP(mMaxLength, hP) * ROUND_UP(mHeadDim, lP) * mBytes;
    }
    if (mQuantValue) {
        mCurrentValueSizePerHead = UP_DIV(mMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP8) * ROUND_UP(mFlashAttentionUpperKv, lP8) + 2 * QUANT_INFO_BYTES * mConfig.mBlockNum * ROUND_UP(mHeadDim, hP8));
    } else {
        mCurrentValueSizePerHead = UP_DIV(mMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP) * ROUND_UP(mFlashAttentionUpperKv, lP) * mBytes);
    }
    size_t keySize = (size_t)mKvNumHead * mCurrentKeySizePerHead;
    size_t valueSize = (size_t)mKvNumHead * mCurrentValueSizePerHead;

    // 2. allocate buffer

    // case1: key&value size exceeds the limited size
    // case2: multi prompts share a common prefix kv cache info
    bool storeKvInDisk  = !mConfig.mKVCacheDir.empty();
    bool sharePrefixKv = mMeta != nullptr && mMeta->file_name.size() > 0 && mMeta->file_flag == KVMeta::PendingWrite;

    if (sharePrefixKv) {
        mSaveShareKvPrefix = true;
        if(!MNNCreateDir(mConfig.mPrefixCacheDir.c_str())) {
            MNN_PRINT("Failed to create prefix cache file dir: %s\n", mConfig.mPrefixCacheDir.c_str());
        }
    }
    if (storeKvInDisk || sharePrefixKv) { // store kv in disk
        std::string keyStoredDst = "";
        std::string valueStoredDst = "";
        if(mMeta != nullptr) {
            mBasePrefixFileName = MNNFilePathConcat(mConfig.mPrefixCacheDir, mMeta->file_name) + "_" + std::to_string(mMeta->layer_index);
            keyStoredDst = sharePrefixKv ? mBasePrefixFileName + ".k" : "";
            valueStoredDst = sharePrefixKv ? mBasePrefixFileName + ".v" : "";
            mMeta->layer_index++;
            mMeta->layer_index = mMeta->layer_index % mMeta->layer_nums;
        }
        createKVCacheFile(keyStoredDst, valueStoredDst);
        resetKVCacheFileSize(keySize, valueSize);
        mmapKVCache(keySize, valueSize);
        mKVCacheInDisk = true;
    } else { // store kv in memory
        mPastKey.reset(Tensor::createDevice<int8_t>({mKvNumHead, (int)mCurrentKeySizePerHead}));
        mPastValue.reset(Tensor::createDevice<int8_t>({mKvNumHead, (int)mCurrentValueSizePerHead}));

        mBackend->onAcquireBuffer(mPastKey.get(), Backend::STATIC);
        mBackend->onAcquireBuffer(mPastValue.get(), Backend::STATIC);

        // initilize 0
        if ((mHeadDim % lP && !mQuantKey) || mQuantKey) {
            memset(mPastKey->host<int8_t>(), 0, mPastKey->length(0) * mPastKey->stride(0));
        }
        if (lP > 1 || mQuantValue) { // can't be mMaxLenth % lP, since mMaxLength may be larger than seq_len for prefilling, we should ensure the (mMaxLength - seq_len)'s buffer is 0.
            memset(mPastValue->host<int8_t>(), 0, mPastValue->length(0) * mPastValue->stride(0));
        }
    }
    // scale, zero point and sum of key for quantization
    if (mQuantKey) { // quant K
        mKeySum.reset(Tensor::createDevice<int8_t>({mKvNumHead, ROUND_UP(mMaxLength, hP8) * QUANT_INFO_BYTES}));
        mKeyMax.reset(Tensor::createDevice<int8_t>({mKvNumHead, mHeadDim * QUANT_INFO_BYTES}));
        mBackend->onAcquireBuffer(mKeySum.get(), Backend::STATIC);
        mBackend->onAcquireBuffer(mKeyMax.get(), Backend::STATIC);

        for (int ks = 0; ks < mKvNumHead * mHeadDim; ++ks) {
            mKeyMax->host<float>()[ks] = std::numeric_limits<float>::lowest();
        }
        if (mBytes == 2) {
            auto core = static_cast<CPUBackend*>(mBackend)->functions();
            core->MNNFp32ToLowp(mKeyMax->host<float>(), (int16_t*)(mKeyMax->host<float>()), mKvNumHead * mHeadDim);
        }
    }
    if (mQuantValue) {
        mValueSum.reset(Tensor::createDevice<int8_t>({mKvNumHead, (int)UP_DIV(mMaxLength, mFlashAttentionUpperKv), ROUND_UP(mHeadDim, hP8) * QUANT_INFO_BYTES}));
        mBackend->onAcquireBuffer(mValueSum.get(), Backend::STATIC);
        memset(mValueSum->host<int8_t>(), 0, mValueSum->stride(0) * mValueSum->length(0));
    }
}

void CPUKVCacheManager::onRealloc(KVMeta* meta) {
    auto kv_seq_len = meta->previous + meta->add - meta->remove + meta->computeReverseSize();
    if (kv_seq_len > mMaxLength) {
        // Realloc
        int oldMaxLength = mMaxLength;
        mMaxLength = (int)kv_seq_len + mConfig.mExpandChunk;
        if (mUseFlashAttention) {
            setFlashAttentionUpperKv(MNN_FLASH_ATTENTION_BLOCK_SIZE);
        } else {
            setFlashAttentionUpperKv(mMaxLength);
        }
        size_t oldKeySize = (size_t)mKvNumHead * mCurrentKeySizePerHead;
        size_t oldValueSize = (size_t)mKvNumHead * mCurrentValueSizePerHead;

        // update current key size per head
        if (mQuantKey) {
            mCurrentKeySizePerHead = ROUND_UP(mMaxLength, hP8) * ROUND_UP(mHeadDim, lP8) + 2 * QUANT_INFO_BYTES * mConfig.mBlockNum * ROUND_UP(mMaxLength, hP8);
        } else {
            mCurrentKeySizePerHead = UP_DIV(mMaxLength, hP) * ROUND_UP(mHeadDim, lP) * hP * mBytes;
        }
        // update current value size per head
        if (mQuantValue) {
            mCurrentValueSizePerHead = UP_DIV(mMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP8) * ROUND_UP(mFlashAttentionUpperKv, lP8) + 2 * QUANT_INFO_BYTES * mConfig.mBlockNum * ROUND_UP(mHeadDim, hP8));
        } else {
            mCurrentValueSizePerHead = UP_DIV(mMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP) * ROUND_UP(mFlashAttentionUpperKv, lP) * mBytes);
        }
        size_t keySize = (size_t)mKvNumHead * mCurrentKeySizePerHead;
        size_t valueSize = (size_t)mKvNumHead * mCurrentValueSizePerHead;

        /*==== No limit for kvcache ====*/
        if (mKVCacheInDisk == false) {
            expandKVCacheInMem(oldMaxLength);
        } else {
            expandKVCacheInDisk(oldMaxLength, oldKeySize, oldValueSize, keySize, valueSize);
        }
        /* No matter where is the kvcache, the scales and zero points are always in memory, since their size is very small */
        if (mQuantKey) {
            auto newKeySumTensor = Tensor::createDevice<int32_t>({mKvNumHead, UP_DIV(mMaxLength, hP8), hP8});
            mBackend->onAcquireBuffer(newKeySumTensor, Backend::STATIC);
            for (int h = 0; h < mKvNumHead; h++) {
                memcpy(newKeySumTensor->host<int8_t>() + h * UP_DIV(mMaxLength, hP8) * hP8 * 4, mKeySum->host<int8_t>() + h * UP_DIV(oldMaxLength, hP8) * hP8 * 4, UP_DIV(oldMaxLength, hP8) * hP8 * 4);
            }
            mKeySum.reset(newKeySumTensor);
        }
        if (mQuantValue) {
            auto newValueSumTensor = Tensor::createDevice<int8_t>({mKvNumHead, (int)UP_DIV(mMaxLength, mFlashAttentionUpperKv), ROUND_UP(mHeadDim, hP8) * QUANT_INFO_BYTES});
            mBackend->onAcquireBuffer(newValueSumTensor, Backend::STATIC);
            auto remainSizePerHead = mValueSum->stride(0);
            auto increSizePerHead = newValueSumTensor->stride(0) - mValueSum->stride(0);
            for (int h = 0; h < mKvNumHead; ++h) {
                memcpy(newValueSumTensor->host<int8_t>() + h * newValueSumTensor->stride(0) , mValueSum->host<int8_t>() + h * mValueSum->stride(0), remainSizePerHead);
                // memset 0
                if (increSizePerHead > 0) {
                    memset(newValueSumTensor->host<int8_t>() + h * newValueSumTensor->stride(0) + remainSizePerHead, 0, increSizePerHead);
                }
            }
            mValueSum.reset(newValueSumTensor);
        }
    }
    // Remove
    auto start = mPastLength - meta->remove;
    if (0 == meta->n_reserve || mQuantKey || mQuantValue) { // n_reserve > 0 is not currently supported when K or V is quantized.
        mPastLength = start;
        return;
    }
#if 1
    auto dstIndex = start;
    for (int n = 0; n < meta->n_reserve; ++n) {
        auto begin = meta->reserve[2 * n];
        auto size  = meta->reserve[2 * n + 1];
        auto srcIndex = start + begin;
        if (mBytes == 2) {
            moveKV<FLOAT16_T>(srcIndex, dstIndex, size);
        } else {
            moveKV<float>(srcIndex, dstIndex, size);
        }
        dstIndex += size;
    }
    mPastLength = dstIndex;
#else
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
        auto keyStride = UP_DIV(mMaxLength, align) * align * ROUND_UP(mHeadDim, lP);
        auto dstKAddr = keyAddr() + dstStartAlign * ROUND_UP(mHeadDim, lP) * mBytes;
        auto srcKAddr = keyAddr() + startAlign * ROUND_UP(mHeadDim, lP) * mBytes;
        for (int i=0; i<mKvNumHead; ++i) {
            auto dst = dstKAddr + i * keyStride * mBytes;
            auto src = srcKAddr + i * keyStride * mBytes;
            for (int j=0; j<sizeUnit; ++j) {
                ::memcpy(dst + j * align * ROUND_UP(mHeadDim, lP) * mBytes, src + j * align * ROUND_UP(mHeadDim, lP) * mBytes, align * ROUND_UP(mHeadDim, lP) * mBytes);
            }
        }


        // Move V
        auto dstVAddr = valudAddr() + dstStartAlign * align * mBytes;
        auto srcVAddr = valudAddr() + startAlign * align * mBytes;
        auto number = mKvNumHead * UP_DIV(mHeadDim, align);
        for (int i=0; i<number; ++i) {
            auto dst = dstVAddr + i * ROUND_UP(mMaxLength, lP) * align * mBytes;
            auto src = srcVAddr + i * ROUND_UP(mMaxLength, lP) * align * mBytes;
            for (int j=0; j<sizeUnit; ++j) {
                ::memcpy(dst + j * align * align * mBytes, src + j * align * align * mBytes, align * align * mBytes);
            }
        }
        dstStart = dstStart + size;
        lastValidSrcEnd = begin + start + size;
    }
    mPastLength = dstStart;
#endif
}

void CPUKVCacheManager::saveKVCacheInDisk() {
    // get original kv cache info
    auto keySize = MNNGetFileSize(mKeyCacheFD);
    auto valueSize = MNNGetFileSize(mValueCacheFD);
    mmapKVCache(keySize, valueSize);
    if(!MNNCreateDir(mConfig.mPrefixCacheDir.c_str())) {
        MNN_PRINT("Failed to create prefix cache file dir: %s\n", mConfig.mPrefixCacheDir.c_str());
    }

    // create new files
    std::string pathk    = MNNFilePathConcat(mConfig.mPrefixCacheDir, mMeta->file_name) + "_" + std::to_string(mMeta->layer_index) + ".k";
    std::string pathv    = MNNFilePathConcat(mConfig.mPrefixCacheDir, mMeta->file_name) + "_" + std::to_string(mMeta->layer_index++) + ".v";
    mMeta->layer_index = mMeta->layer_index % mMeta->layer_nums;

    auto new_key_fd   = MNNCreateFile(pathk.c_str());
    auto new_value_fd = MNNCreateFile(pathv.c_str());
    if (new_key_fd == INVALID_FILE) {
        MNN_PRINT("Failed to create the file: %s\n", pathk.c_str());
    }
    if (new_value_fd == INVALID_FILE) {
        MNN_PRINT("Failed to create the file: %s\n", pathv.c_str());
    }
    // set new file size
    if (MNNSetFileSize(new_key_fd, keySize) != MNN::NO_ERROR || MNNSetFileSize(new_value_fd, valueSize) != MNN::NO_ERROR) {
        MNN_PRINT("Failed to resize the kvcache files!\n");
    }
    // mmap files
    int8_t* mMapNewKeyAddr = (int8_t *)MNNMmapFile(new_key_fd, keySize);
    if (mMapNewKeyAddr == nullptr) {
        MNN_PRINT("Failed to memory-map the new kvcache!\n");
    }
    int8_t* mMapNewValueAddr =(int8_t *)MNNMmapFile(new_value_fd, valueSize);
    if (mMapNewValueAddr == nullptr) {
        MNN_PRINT("Failed to memory-map the kvcache!\n");
    }

    // copy
    memcpy(mMapNewKeyAddr,   mMapKeyAddr,   keySize);
    memcpy(mMapNewValueAddr, mMapValueAddr, valueSize);

    // unmap new files
    if (mMapNewKeyAddr != nullptr) {
        MNNUnmapFile(mMapNewKeyAddr, keySize);
        mMapNewKeyAddr = nullptr;
    }
    if (mMapNewValueAddr != nullptr) {
        MNNUnmapFile(mMapNewValueAddr, valueSize);
        mMapNewValueAddr = nullptr;
    }
    // close file
    if (new_key_fd != INVALID_FILE) {
        MNNCloseFile(new_key_fd);
        new_key_fd = INVALID_FILE;
    }
    if (new_value_fd != INVALID_FILE) {
        MNNCloseFile(new_value_fd);
        new_value_fd = INVALID_FILE;
    }
}

void CPUKVCacheManager::onClear() {
    if (mKVCacheInDisk) {
        // mSaveShareKvPrefix also need unmap file
        unmapKVCache(mCurrentKeySizePerHead * (size_t)mKvNumHead, mCurrentValueSizePerHead * (size_t)mKvNumHead);
        if(mSaveShareKvPrefix) {
            // set prefix cachefile validation
            auto k_file = mBasePrefixFileName + ".k";
            if(MNNFileExist(k_file.c_str())) {
                auto k_sync_file = mBasePrefixFileName + "_sync.k";
                MNNCreateFile(k_sync_file.c_str());
            }
            auto v_file = mBasePrefixFileName + ".v";
            if(MNNFileExist(v_file.c_str())) {
                auto v_sync_file = mBasePrefixFileName + "_sync.v";
                MNNCreateFile(v_sync_file.c_str());
            }
        } else {
            // delete temp kvcache file
            removeKVCacheFile();
        }
        mKVCacheInDisk = false;
    }
    mPastKey.reset();
    mPastValue.reset();
    mKeySum.reset();
    mKeyMax.reset();
    mValueSum.reset();
    mMaxLength = mPastLength = 0;
}

template <typename T>
void CPUKVCacheManager::ProcessKey(const Tensor* key, int seqLen, int kvHead) {
    if (mQuantKey) {  // [seqLen, headDim] -> [maxlen/hP8, blockNum, (headDim/blockNum)/lP8, hP8, lP8]
        int8_t * keyDst = reinterpret_cast<int8_t*>(addrOfKey(kvHead));
        float * sumDst = reinterpret_cast<float*>(addrOfKeySum(kvHead));

        auto blockL = UP_DIV(mHeadDim, mConfig.mBlockNum);
        auto weightStride1 = ROUND_UP(blockL, lP8) * hP8;
        auto weightStride2 = lP8 * hP8;
        auto packedWeightStride1 = weightStride1 + 2 * QUANT_INFO_BYTES * hP8;

        T* keyMax = reinterpret_cast<T*>(addrOfKeyMax(kvHead));
        int32_t params[] = {mKvNumHead, seqLen, mHeadDim, mConfig.mBlockNum, eP8, lP8, hP8, mPastLength, kvHead};
        mQuantKeyFunc(keyDst, key->host<float>(), sumDst, (float*)keyMax, params);
    }
    else { // target: [maxlen/hP, headdim/lP, hP, lP]
        T * key_dst = reinterpret_cast<T*>(addrOfKey(kvHead));
        auto stride0 = ROUND_UP(mHeadDim, lP) * hP;
        auto stride1 = hP * lP;
        for (int i = 0; i < seqLen; i++) {
            T * key_src = key->host<T>() + i * mKvNumHead * mHeadDim + kvHead * mHeadDim;
            int out_index = (mPastLength + i) / hP;
            int in_index  = (mPastLength + i) % hP;
            for (int j = 0; j < mHeadDim; j++) {
                key_dst[out_index * stride0 + (j / lP) * stride1 + in_index * lP + (j % lP)] = key_src[j];
            }
        }
    }
}

template <typename T>
void CPUKVCacheManager::ProcessValue(const Tensor* value, int seqLen, int kvHead) { // [headdim/hP, maxlen, hP]
    if (mQuantValue) {
        int8_t* valueDst = reinterpret_cast<int8_t*>(addrOfValue(kvHead));
        float* valueSum = reinterpret_cast<float*>(addrOfValueSum(kvHead));

        int32_t params[] = {mKvNumHead, seqLen, mHeadDim, mConfig.mBlockNum, mMaxLength, lP8, hP8, mPastLength, kvHead, (int32_t)mFlashAttentionUpperKv};
        mQuantValueFunc(valueDst, value->host<float>(), valueSum, params);
    }
    else {
        // [mHeadDim/hP, mMaxLength/lP, hP, lP]
        auto stride0 = ROUND_UP(mMaxLength, lP) * hP;
        auto stride1 = hP * lP;

        auto weightStride2 = lP * hP;
        auto weightStride1 = UP_DIV((int32_t)mFlashAttentionUpperKv, lP) * weightStride2;
        auto weightStride0 = weightStride1 * UP_DIV(mHeadDim, hP);

        T * value_dst = reinterpret_cast<T*>(addrOfValue(kvHead));
        for (int i = 0; i < seqLen; i++) {
            T * value_src = value->host<T>() + i * mKvNumHead * mHeadDim + kvHead * mHeadDim;
            // int seqLenOut = (mPastLength + i) / lP;
            // int seqLenIn = (mPastLength + i) % lP;

            int kvSeqIndx = mPastLength + i;
            int idxInner = (kvSeqIndx / (int32_t)mFlashAttentionUpperKv) * weightStride0 + (kvSeqIndx % (int32_t)mFlashAttentionUpperKv) / lP * weightStride2 + (kvSeqIndx % (int32_t)mFlashAttentionUpperKv) % lP;
            for (int j = 0; j < mHeadDim; j++) {
                int idxBase = (j / hP) * weightStride1 + (j % hP) * lP;
                int out_index = j / hP;
                int in_index  = j % hP;
                // value_dst[out_index * stride0 + seqLenOut * stride1 + in_index * lP + seqLenIn] = value_src[j];
                value_dst[idxBase + idxInner] = value_src[j];
            }
        }
    }
}

size_t CPUKVCacheManager::keyIndex(int seq, int dim) const {
    return (seq / hP) * ROUND_UP(mHeadDim, lP) * hP +
           (dim / lP) * hP * lP +
           (seq % hP) * lP +
           (dim % lP);
}

size_t CPUKVCacheManager::valueIndex(int seq, int dim) const {
    return (dim / hP) * ROUND_UP(mMaxLength, lP) * hP +
           (seq / lP) * hP * lP +
           (dim % hP) * lP +
           (seq % lP);
}

template <typename T>
void CPUKVCacheManager::moveKV(int src, int dst, int size) {
    for (int h = 0; h < mKvNumHead; ++h) {
        auto kPtr = reinterpret_cast<T*>(addrOfKey(h));
        auto vPtr = reinterpret_cast<T*>(addrOfValue(h));
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < mHeadDim; j++) {
                kPtr[keyIndex(dst + i, j)]   = kPtr[keyIndex(src + i, j)];
                vPtr[valueIndex(dst + i, j)] = vPtr[valueIndex(src + i, j)];
            }
        }
    }
}

void CPUKVCacheManager::onUpdateKV(const Tensor * key, const Tensor * value, int add) {
    auto core = static_cast<CPUBackend*>(mBackend)->functions();
    int seq_len = add;
    auto divPart = UP_DIV(mKvNumHead, 1);
    MNN_CONCURRENCY_BEGIN(tId, 1) {
        auto remainPart = mKvNumHead - tId * divPart;
        if (remainPart > 0) {
            remainPart = ALIMIN(divPart, remainPart);
            int startIdx = tId * divPart;
            int endIdx = startIdx + remainPart;
            for (int h = startIdx; h < endIdx; ++h) {
                if (mBytes == 2) {
                    ProcessKey<FLOAT16_T>(key, seq_len, h);
                    ProcessValue<FLOAT16_T>(value, seq_len, h);
                } else {
                    ProcessKey<float>(key, seq_len, h);
                    ProcessValue<float>(value, seq_len, h);
                }
            }
        }
    } MNN_CONCURRENCY_END();
    mPastLength += seq_len;
}

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
