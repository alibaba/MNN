//
//  MetalKVCacheManager.mm
//  MNN
//
//  Created by MNN on 2025/12/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#import "backend/metal/MetalBackend.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "MetalKVCacheManager.hpp"
#import "MetalExecution.hpp"
#include <algorithm>
#include <memory>

namespace MNN {

void MetalKVCacheManager::addReplayUser(MetalExecution* execution) {
    mReplayUsers.emplace_back(execution);
}

void MetalKVCacheManager::removeReplayUser(MetalExecution* execution) {
    auto iter = std::find(mReplayUsers.begin(), mReplayUsers.end(), execution);
    if (iter != mReplayUsers.end()) {
        mReplayUsers.erase(iter);
    }
}

void MetalKVCacheManager::invalidateReplay() {
    for (auto execution : mReplayUsers) {
        execution->invalidateReplay();
    }
}

void MetalKVCacheManager::onResize(int kv_num_head, int head_dim) {
    mKvNumHead = kv_num_head;
    mHeadDim = head_dim;
    auto mtbn = static_cast<MetalBackend *>(mBackend);
    // Record bytes for K cache element. When mQuantKey is enabled, key is stored as int8.
    mBytes = mQuantKey ? 1 : (mtbn->useFp16InsteadFp32() ? 2 : 4);
}

ErrorCode MetalKVCacheManager::onAlloc(KVMeta* meta, int seq_len) {
    mMeta = meta;
    auto mtbn = static_cast<MetalBackend *>(mBackend);
    auto context = (__bridge MNNMetalContext *)mtbn->context();

    auto kv_seq_len = mMeta != nullptr ? mMeta->add : seq_len;
    int keyByte = mQuantKey ? 1 : (mtbn->useFp16InsteadFp32() ? 2 : 4);
    int valueByte = mQuantValue ? 1 : (mtbn->useFp16InsteadFp32() ? 2 : 4);
    // load disk prefix kvcache
    if(mMeta != nullptr && mMeta->file_name.size() > 0 && mMeta->file_flag == KVMeta::PendingRead) {
        // create new files
        std::string pathk    = MNNFilePathConcat(mConfig.mPrefixCacheDir, mMeta->file_name) + "_" + std::to_string(mMeta->layer_index) + ".k";
        std::string pathv    = MNNFilePathConcat(mConfig.mPrefixCacheDir, mMeta->file_name) + "_" + std::to_string(mMeta->layer_index++) + ".v";
        mMeta->layer_index = mMeta->layer_index % mMeta->layer_nums;
        auto old_key_fd   = MNNOpenFile(pathk.c_str(), MNN_FILE_READ | MNN_FILE_WRITE);
        auto old_value_fd = MNNOpenFile(pathv.c_str(), MNN_FILE_READ | MNN_FILE_WRITE);
        if (old_key_fd == INVALID_FILE) {
            MNN_PRINT("Failed to open the file: %s\n", pathk.c_str());
        }
        if (old_value_fd == INVALID_FILE) {
            MNN_PRINT("Failed to open the file: %s\n", pathv.c_str());
        }

        // get kv cache file info
        auto oldKeySize = MNNGetFileSize(old_key_fd);
        auto oldValueSize = MNNGetFileSize(old_value_fd);
        const size_t keyRowBytes = (size_t)mKvNumHead * mHeadDim * keyByte;
        const size_t valueRowBytes = (size_t)mKvNumHead * mHeadDim * valueByte;
        size_t oldKeyMaxLength = oldKeySize / keyRowBytes;
        size_t oldValueMaxLength = oldValueSize / valueRowBytes;
        size_t oldMaxLength = ALIMIN(oldKeyMaxLength, oldValueMaxLength);
        if(oldMaxLength < meta->seqlen_in_disk) {
            MNN_ERROR("[Error]: Kvcache in disk size smaller than saved lengthInDiskToload:%d\n", (int)meta->seqlen_in_disk);
        }

        int kv_seq_len = ROUND_UP(meta->add + meta->seqlen_in_disk, mConfig.mKvAlignNum);
        mMaxLength = kv_seq_len > oldMaxLength ? ROUND_UP(meta->add + meta->seqlen_in_disk + mConfig.mExpandChunk, mConfig.mKvAlignNum) : oldMaxLength;
        mCurrentKeySize = ALIMAX((size_t)mKvNumHead * mMaxLength * mHeadDim * keyByte, (size_t)oldKeySize);
        mCurrentValueSize = ALIMAX((size_t)mKvNumHead * mMaxLength * mHeadDim * valueByte, (size_t)oldValueSize);

        createKVCacheFile();
        resetKVCacheFileSize(mCurrentKeySize, mCurrentValueSize);
        expandKVCacheInDisk(oldKeySize, oldValueSize, mCurrentKeySize, mCurrentValueSize,
                            (int)meta->seqlen_in_disk, true, old_key_fd, old_value_fd);

        mPastLength = meta->seqlen_in_disk;
        mKVCacheInDisk = true;

        return NO_ERROR;
    }

    bool storeKvInDisk = !mConfig.mKVCacheDir.empty();
    bool sharePrefixKv = mMeta != nullptr && mMeta->file_name.size() > 0 && mMeta->file_flag == KVMeta::PendingWrite;
    auto capacity = kv_seq_len + mConfig.mExpandChunk;
    if (!storeKvInDisk && !sharePrefixKv && mMeta != nullptr) {
        capacity = std::max(capacity, mMeta->capacity);
    }
    // align max kv_seq_len to mKvAlignNum, for simd/tensor matrix load alignment
    const int prevMaxLength = mMaxLength;
    const size_t prevKeySize = mCurrentKeySize;
    const size_t prevValueSize = mCurrentValueSize;
    mMaxLength = ROUND_UP(capacity, mConfig.mKvAlignNum);
    size_t totalSize = (size_t)mKvNumHead * mMaxLength * mHeadDim * keyByte;
    size_t valueTotalSize = (size_t)mKvNumHead * mMaxLength * mHeadDim * valueByte;
    mCurrentKeySize = totalSize;
    mCurrentValueSize = valueTotalSize;

    if (sharePrefixKv) {
        mSaveShareKvPrefix = true;
        if(!MNNCreateDir(mConfig.mPrefixCacheDir.c_str())) {
            MNN_PRINT("Failed to create prefix cache file dir: %s\n", mConfig.mPrefixCacheDir.c_str());
        }
    }

    if(storeKvInDisk || sharePrefixKv) {
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
        resetKVCacheFileSize(totalSize, valueTotalSize);
        mmapKVCache(totalSize, valueTotalSize);
        mKVCacheInDisk = true;
        mKeyBuffer   = [[context device] newBufferWithBytesNoCopy:mMapKeyAddr length:totalSize options:MTLResourceStorageModeShared  deallocator:nil];
        mValueBuffer = [[context device] newBufferWithBytesNoCopy:mMapValueAddr length:valueTotalSize options:MTLResourceStorageModeShared  deallocator:nil];
        if (useDynamicScaleBuffer()) {
            int scaleByte = mtbn->useFp16InsteadFp32() ? 2 : 4;
            mKScaleBuffer = [[context device] newBufferWithLength:mMaxLength * scaleByte * 2 options:MTLResourceStorageModeShared];
            mVScaleBuffer = [[context device] newBufferWithLength:mMaxLength * scaleByte * 2 options:MTLResourceStorageModeShared];
            // Zero the whole scale/bias tail: dequant(0) = bias, so the K/V
            // memset above only makes padded tokens vanish if bias is 0 too.
            // Uninitialized bias bits can decode as Inf/NaN and 0*NaN = NaN
            // poisons the AV matmul for recycled (non-zero) pages.
            ::memset([mKScaleBuffer contents], 0, mMaxLength * scaleByte * 2);
            ::memset([mVScaleBuffer contents], 0, mMaxLength * scaleByte * 2);
        } else {
            mKScaleBuffer = nil;
            mVScaleBuffer = nil;
        }
        auto new_key_ptr = (uint8_t*)[mKeyBuffer contents];
        ::memset(new_key_ptr, 0, totalSize);
        auto new_value_ptr = (uint8_t*)[mValueBuffer contents];
        ::memset(new_value_ptr, 0, valueTotalSize);
    } else {
        // past_key: [maxlen, kvNumhead, headdim]
        // past_value: [maxlen, kvNumhead, headdim] (row-major, same as key)
        // The reservation is a target, not a requirement: fall back to the length this
        // forward needs rather than memset-ing a nil buffer.
        const int reservedLength = mMaxLength;
        const int neededLength = ROUND_UP(kv_seq_len + mConfig.mExpandChunk, mConfig.mKvAlignNum);
        std::unique_ptr<Tensor> new_key, new_value;
        auto acquireKV = [&](int length) {
            new_key.reset(Tensor::createDevice<int8_t>({length, mKvNumHead, mHeadDim * keyByte}));
            new_value.reset(Tensor::createDevice<int8_t>({length, mKvNumHead, mHeadDim * valueByte}));
            const bool keyAcquired = mBackend->onAcquireBuffer(new_key.get(), Backend::STATIC);
            const bool valueAcquired = keyAcquired && mBackend->onAcquireBuffer(new_value.get(), Backend::STATIC);
            if (valueAcquired && nil != MetalBackend::getBuffer(new_key.get()).first &&
                nil != MetalBackend::getBuffer(new_value.get()).first) {
                mMaxLength = length;
                return true;
            }
            if (valueAcquired) {
                mBackend->onReleaseBuffer(new_value.get(), Backend::STATIC);
            }
            if (keyAcquired) {
                mBackend->onReleaseBuffer(new_key.get(), Backend::STATIC);
            }
            return false;
        };
        if (!acquireKV(reservedLength) && (reservedLength == neededLength || !acquireKV(neededLength))) {
            // No buffer was committed, so the capacity claim must not outlive the attempt.
            mMaxLength = prevMaxLength;
            mCurrentKeySize = prevKeySize;
            mCurrentValueSize = prevValueSize;
            return OUT_OF_MEMORY;
        }
        mCurrentKeySize = (size_t)mKvNumHead * mMaxLength * mHeadDim * keyByte;
        mCurrentValueSize = (size_t)mKvNumHead * mMaxLength * mHeadDim * valueByte;
        // memset for qkv matmul mad, in case dirty data
        auto newKeyBuf = MetalBackend::getBuffer(new_key.get());
        auto new_key_ptr = (uint8_t*)[newKeyBuf.first contents] + newKeyBuf.second;
        ::memset(new_key_ptr, 0, mCurrentKeySize);
        auto newValueBuf = MetalBackend::getBuffer(new_value.get());
        auto new_value_ptr = (uint8_t*)[newValueBuf.first contents] + newValueBuf.second;
        ::memset(new_value_ptr, 0, (size_t)mKvNumHead * mMaxLength * mHeadDim * valueByte);
        mPastKey.reset(new_key.release());
        mPastValue.reset(new_value.release());
        if (useDynamicScaleBuffer()) {
            int scaleByte = mtbn->useFp16InsteadFp32() ? 2 : 4;
            mKScaleBuffer = [[context device] newBufferWithLength:mMaxLength * scaleByte * 2 options:MTLResourceStorageModeShared];
            mVScaleBuffer = [[context device] newBufferWithLength:mMaxLength * scaleByte * 2 options:MTLResourceStorageModeShared];
            // See the in-disk branch above: zero scale+bias so padded tokens
            // dequantize to 0 instead of uninitialized (possibly NaN) bias.
            ::memset([mKScaleBuffer contents], 0, mMaxLength * scaleByte * 2);
            ::memset([mVScaleBuffer contents], 0, mMaxLength * scaleByte * 2);
        } else {
            mKScaleBuffer = nil;
            mVScaleBuffer = nil;
        }

    }
    return NO_ERROR;
}
ErrorCode MetalKVCacheManager::onRealloc(KVMeta* meta) {
    mMeta = meta;
    auto kv_seq_len = mMeta->previous + mMeta->add - mMeta->remove + mMeta->computeReverseSize();
    auto mtbn = static_cast<MetalBackend *>(mBackend);

    int keyByte = mQuantKey ? 1 : (mtbn->useFp16InsteadFp32() ? 2 : 4);
    int valueByte = mQuantValue ? 1 : (mtbn->useFp16InsteadFp32() ? 2 : 4);

    auto start = mMeta->previous - mMeta->remove;
    auto capacity = mKVCacheInDisk ? kv_seq_len : std::max(kv_seq_len, mMeta->capacity);
    // latest length larger than maxLen
    if (capacity > mMaxLength) {
        invalidateReplay();

        // copy mPastLength including all remove/reverse to new buffer first
        auto copy_len = mPastLength;
        bool needCopy = mPastLength > 0;

        size_t old_size = (size_t)mKvNumHead * copy_len * mHeadDim * keyByte;

        auto oldKeyTotalSize = mCurrentKeySize;
        auto oldValueTotalSize = mCurrentValueSize;
        auto oldMaxLength = mMaxLength;
        // align max kv_seq_len to mKvAlignNum, for simd/tensor matrix load alignment
        capacity = std::max(capacity, kv_seq_len + mConfig.mExpandChunk);
        mMaxLength = ROUND_UP(capacity, mConfig.mKvAlignNum);

        mCurrentKeySize = (size_t)mKvNumHead * mMaxLength * mHeadDim * keyByte;
        mCurrentValueSize = (size_t)mKvNumHead * mMaxLength * mHeadDim * valueByte;

        mPastLength = (int)start;

        if(mKVCacheInDisk) {
            expandKVCacheInDisk(oldKeyTotalSize, oldValueTotalSize, mCurrentKeySize, mCurrentValueSize,
                                (int)copy_len, needCopy);
        } else if (!expandKVCacheInMem(old_size, (int)copy_len, needCopy)) {
            // A failed reservation must not leave the cache smaller than the tokens this
            // forward writes, so retry at the organic need before keeping the old cache.
            const int neededLength = ROUND_UP(kv_seq_len + mConfig.mExpandChunk, mConfig.mKvAlignNum);
            bool retryOk = false;
            if (neededLength < mMaxLength) {
                mMaxLength = neededLength;
                mCurrentKeySize = (size_t)mKvNumHead * mMaxLength * mHeadDim * keyByte;
                mCurrentValueSize = (size_t)mKvNumHead * mMaxLength * mHeadDim * valueByte;
                retryOk = expandKVCacheInMem(old_size, (int)copy_len, needCopy);
            }
            if (!retryOk) {
                mMaxLength = oldMaxLength;
                mCurrentKeySize = oldKeyTotalSize;
                mCurrentValueSize = oldValueTotalSize;
                return OUT_OF_MEMORY;
            }
        }
    }

    // Remove
    {
        if (0 == mMeta->n_reserve) {
            mPastLength = start;
            return NO_ERROR;
        }

        int8_t *key_ptr = nullptr;
        int8_t *value_ptr = nullptr;
        if(mKVCacheInDisk) {
            key_ptr = mMapKeyAddr;
            value_ptr = mMapValueAddr;
        } else {
            auto keyBuf = MetalBackend::getBuffer(mPastKey.get());
            key_ptr = (int8_t*)[keyBuf.first contents] + keyBuf.second;
            auto valueBuf = MetalBackend::getBuffer(mPastValue.get());
            value_ptr = (int8_t*)[valueBuf.first contents] + valueBuf.second;
        }
        auto src_start = start;
        // TODO: need to ensure reserve info is sorted
        for (int n = 0; n < mMeta->n_reserve; ++n) {
            auto begin = mMeta->reserve[2 * n];
            auto length = mMeta->reserve[2 * n + 1];
            // past_key   : [mCache->mPastLength, mKvNumHead, mHeadDim]
            // past_value : [mCache->mPastLength, mKvNumHead, mHeadDim] (row-major, same as key)

            auto copy_src_index = src_start + begin;
            auto copy_dst_index = start;
            for(int i = 0; i < length; i++) {
                ::memcpy(key_ptr + (copy_dst_index + i) * mKvNumHead * mHeadDim * keyByte, key_ptr + (copy_src_index + i) * mKvNumHead * mHeadDim * keyByte, mKvNumHead * mHeadDim * keyByte);
            }
            for(int i = 0; i < length; i++) {
                ::memcpy(value_ptr + (copy_dst_index + i) * mKvNumHead * mHeadDim * valueByte, value_ptr + (copy_src_index + i) * mKvNumHead * mHeadDim * valueByte, mKvNumHead * mHeadDim * valueByte);
            }
            if (mKScaleBuffer != nil) {
                int scaleByte = mtbn->useFp16InsteadFp32() ? 2 : 4;
                if (scaleByte == 2) {
                    int16_t* k_scale_ptr = (int16_t*)[mKScaleBuffer contents];
                    int16_t* v_scale_ptr = (int16_t*)[mVScaleBuffer contents];
                    for(int i = 0; i < length; i++) {
                        k_scale_ptr[(copy_dst_index + i) * 2 + 0] = k_scale_ptr[(copy_src_index + i) * 2 + 0];
                        k_scale_ptr[(copy_dst_index + i) * 2 + 1] = k_scale_ptr[(copy_src_index + i) * 2 + 1];
                        v_scale_ptr[(copy_dst_index + i) * 2 + 0] = v_scale_ptr[(copy_src_index + i) * 2 + 0];
                        v_scale_ptr[(copy_dst_index + i) * 2 + 1] = v_scale_ptr[(copy_src_index + i) * 2 + 1];
                    }
                } else {
                    float* k_scale_ptr = (float*)[mKScaleBuffer contents];
                    float* v_scale_ptr = (float*)[mVScaleBuffer contents];
                    for(int i = 0; i < length; i++) {
                        k_scale_ptr[(copy_dst_index + i) * 2 + 0] = k_scale_ptr[(copy_src_index + i) * 2 + 0];
                        k_scale_ptr[(copy_dst_index + i) * 2 + 1] = k_scale_ptr[(copy_src_index + i) * 2 + 1];
                        v_scale_ptr[(copy_dst_index + i) * 2 + 0] = v_scale_ptr[(copy_src_index + i) * 2 + 0];
                        v_scale_ptr[(copy_dst_index + i) * 2 + 1] = v_scale_ptr[(copy_src_index + i) * 2 + 1];
                    }
                }
            }
            start += length;
        }
        mPastLength = (int)start;
    }
    return NO_ERROR;
}

bool MetalKVCacheManager::expandKVCacheInMem(size_t oldSize, int copy_len, bool need_copy) {
    auto mtbn = static_cast<MetalBackend *>(mBackend);
    int keyByte = mQuantKey ? 1 : (mtbn->useFp16InsteadFp32() ? 2 : 4);
    int valueByte = mQuantValue ? 1 : (mtbn->useFp16InsteadFp32() ? 2 : 4);
    // past_key: [maxlen, kvNumhead, headdim]
    std::unique_ptr<Tensor> new_key(Tensor::createDevice<int8_t>({mMaxLength, mKvNumHead, mHeadDim * keyByte}));
    // past_value: [maxlen, kvNumhead, headdim] (row-major, same as key)
    std::unique_ptr<Tensor> new_value(Tensor::createDevice<int8_t>({mMaxLength, mKvNumHead, mHeadDim * valueByte}));

    const bool keyAcquired = mBackend->onAcquireBuffer(new_key.get(), Backend::STATIC);
    const bool valueAcquired = keyAcquired && mBackend->onAcquireBuffer(new_value.get(), Backend::STATIC);
    auto newKeyBuf = MetalBackend::getBuffer(new_key.get());
    auto newValueBuf = MetalBackend::getBuffer(new_value.get());
    if (!valueAcquired || nil == newKeyBuf.first || nil == newValueBuf.first) {
        // Keep the old cache rather than memset a nil buffer, and return whatever was acquired to the pool.
        if (valueAcquired) {
            mBackend->onReleaseBuffer(new_value.get(), Backend::STATIC);
        }
        if (keyAcquired) {
            mBackend->onReleaseBuffer(new_key.get(), Backend::STATIC);
        }
        return false;
    }

    // memset for qkv matmul mad, in case dirty data
    auto new_key_ptr = (uint8_t*)[newKeyBuf.first contents] + newKeyBuf.second;
    ::memset(new_key_ptr, 0, (size_t)mMaxLength * mKvNumHead * mHeadDim * keyByte);

    auto new_value_ptr = (uint8_t*)[newValueBuf.first contents] + newValueBuf.second;
    ::memset(new_value_ptr, 0, (size_t)mMaxLength * mKvNumHead * mHeadDim * valueByte);

    if (need_copy) {
        auto keyBuf = MetalBackend::getBuffer(mPastKey.get());
        auto key_ptr = (uint8_t*)[keyBuf.first contents] + keyBuf.second;;
        ::memcpy(new_key_ptr, key_ptr, oldSize);

        auto valueBuf = MetalBackend::getBuffer(mPastValue.get());
        auto value_ptr = (uint8_t*)[valueBuf.first contents] + valueBuf.second;
        ::memcpy(new_value_ptr, value_ptr, (size_t)copy_len * mKvNumHead * mHeadDim * valueByte);
    }

    mPastKey.reset(new_key.release());
    mPastValue.reset(new_value.release());

    auto context = (__bridge MNNMetalContext *)mtbn->context();
    if (useDynamicScaleBuffer()) {
        int scaleByte = mtbn->useFp16InsteadFp32() ? 2 : 4;
        id<MTLBuffer> newKScale = [[context device] newBufferWithLength:mMaxLength * scaleByte * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> newVScale = [[context device] newBufferWithLength:mMaxLength * scaleByte * 2 options:MTLResourceStorageModeShared];
        // Zero before the copy so the grown tail past copy_len is not left
        // uninitialized (padded tokens must dequantize to 0, see onAlloc).
        ::memset([newKScale contents], 0, mMaxLength * scaleByte * 2);
        ::memset([newVScale contents], 0, mMaxLength * scaleByte * 2);
        if (need_copy && mKScaleBuffer != nil) {
            ::memcpy([newKScale contents], [mKScaleBuffer contents], (size_t)copy_len * scaleByte * 2);
            ::memcpy([newVScale contents], [mVScaleBuffer contents], (size_t)copy_len * scaleByte * 2);
        }
        mKScaleBuffer = newKScale;
        mVScaleBuffer = newVScale;
    } else {
        mKScaleBuffer = nil;
        mVScaleBuffer = nil;
    }
    return true;
}

void MetalKVCacheManager::expandKVCacheInDisk(size_t oldKeySize, size_t oldValueSize, size_t curKeySize,
                                              size_t curValueSize, int copy_len, bool need_copy,
                                              file_t specKeyFile, file_t specValueFile) {
    auto mtbn = static_cast<MetalBackend *>(mBackend);
    auto context = (__bridge MNNMetalContext *)mtbn->context();

    mmapKVCache(oldKeySize, oldValueSize, specKeyFile, specValueFile);
    std::vector<int8_t> prevKey, prevValue;
    prevKey.resize(oldKeySize);
    prevValue.resize(oldValueSize);
    memcpy(prevKey.data(),   mMapKeyAddr,   oldKeySize);
    memcpy(prevValue.data(), mMapValueAddr, oldValueSize);

    unmapKVCache(oldKeySize, oldValueSize);
    resetKVCacheFileSize(curKeySize, curValueSize);
    mmapKVCache(curKeySize, curValueSize);

    // reset id<MTLBuffer>
    mKeyBuffer   = [[context device] newBufferWithBytesNoCopy:mMapKeyAddr length:curKeySize options:MTLResourceStorageModeShared  deallocator:nil];
    mValueBuffer = [[context device] newBufferWithBytesNoCopy:mMapValueAddr length:curValueSize options:MTLResourceStorageModeShared  deallocator:nil];

    if (useDynamicScaleBuffer()) {
        int scaleByte = mtbn->useFp16InsteadFp32() ? 2 : 4;
        id<MTLBuffer> newKScale = [[context device] newBufferWithLength:mMaxLength * scaleByte * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> newVScale = [[context device] newBufferWithLength:mMaxLength * scaleByte * 2 options:MTLResourceStorageModeShared];
        // Zero before the copy so the grown tail past copy_len is not left
        // uninitialized (padded tokens must dequantize to 0, see onAlloc).
        ::memset([newKScale contents], 0, mMaxLength * scaleByte * 2);
        ::memset([newVScale contents], 0, mMaxLength * scaleByte * 2);
        if (need_copy && mKScaleBuffer != nil) {
            ::memcpy([newKScale contents], [mKScaleBuffer contents], (size_t)copy_len * scaleByte * 2);
            ::memcpy([newVScale contents], [mVScaleBuffer contents], (size_t)copy_len * scaleByte * 2);
        }
        mKScaleBuffer = newKScale;
        mVScaleBuffer = newVScale;
    } else {
        mKScaleBuffer = nil;
        mVScaleBuffer = nil;
    }


    // Step 3: Move the kvcache from temporary buffers in memory to disk
    memset(mMapKeyAddr, 0, curKeySize);
    memset(mMapValueAddr, 0, curValueSize);

    if (need_copy) {
        ::memcpy(mMapKeyAddr, prevKey.data(), oldKeySize);
        ::memcpy(mMapValueAddr, prevValue.data(), oldValueSize);
    }
}

void MetalKVCacheManager::onClear() {
    invalidateReplay();
    if (mKVCacheInDisk) {
        mKeyBuffer = nil;
        mValueBuffer = nil;

        unmapKVCache(mCurrentKeySize, mCurrentValueSize);
        if (!mSaveShareKvPrefix) {
            removeKVCacheFile();
        }
        mKVCacheInDisk = false;
    }
    mPastKey.reset();
    mPastValue.reset();
    mKScaleBuffer = nil;
    mVScaleBuffer = nil;
    mMaxLength = 0;
    mPastLength = 0;
}
} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
