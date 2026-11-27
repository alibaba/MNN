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
#include <new>

namespace MNN {

namespace {
// Prefix files are only read during onAlloc; close them on every exit path, failures included.
struct ScopedPrefixFile {
    file_t fd;
    explicit ScopedPrefixFile(const std::string& path) : fd(MNNOpenFile(path.c_str(), MNN_FILE_READ | MNN_FILE_WRITE)) {
        if (fd == INVALID_FILE) {
            MNN_PRINT("Failed to open the file: %s\n", path.c_str());
        }
    }
    ~ScopedPrefixFile() {
        if (fd != INVALID_FILE) {
            MNNCloseFile(fd);
        }
    }
    ScopedPrefixFile(const ScopedPrefixFile&) = delete;
    ScopedPrefixFile& operator=(const ScopedPrefixFile&) = delete;
};
} // namespace

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
        ScopedPrefixFile oldKeyFile(pathk), oldValueFile(pathv);
        if (oldKeyFile.fd == INVALID_FILE || oldValueFile.fd == INVALID_FILE) {
            return FILE_OPEN_FAILED;
        }

        // get kv cache file info
        auto oldKeySize = MNNGetFileSize(oldKeyFile.fd);
        auto oldValueSize = MNNGetFileSize(oldValueFile.fd);
        if (oldKeySize == INVALID_SIZE || oldValueSize == INVALID_SIZE) {
            return FILE_OPEN_FAILED;
        }
        const size_t keyRowBytes = (size_t)mBatch * mKvNumHead * mHeadDim * keyByte;
        const size_t valueRowBytes = (size_t)mBatch * mKvNumHead * mHeadDim * valueByte;
        size_t oldKeyMaxLength = oldKeySize / keyRowBytes;
        size_t oldValueMaxLength = oldValueSize / valueRowBytes;
        size_t oldMaxLength = ALIMIN(oldKeyMaxLength, oldValueMaxLength);
        if(oldMaxLength < (size_t)meta->seqlen_in_disk) {
            // Loading past the file's end would expose uninitialized rows as history.
            MNN_ERROR("[Error]: Kvcache in disk size smaller than saved lengthInDiskToload:%d\n", (int)meta->seqlen_in_disk);
            return INPUT_DATA_ERROR;
        }

        int kv_seq_len = ROUND_UP(meta->add + meta->seqlen_in_disk, mConfig.mKvAlignNum);
        mMaxLength = kv_seq_len > oldMaxLength ? ROUND_UP(meta->add + meta->seqlen_in_disk + mConfig.mExpandChunk, mConfig.mKvAlignNum) : oldMaxLength;
        mCurrentKeySize = ALIMAX((size_t)mBatch * mKvNumHead * mMaxLength * mHeadDim * keyByte, (size_t)oldKeySize);
        mCurrentValueSize = ALIMAX((size_t)mBatch * mKvNumHead * mMaxLength * mHeadDim * valueByte, (size_t)oldValueSize);

        auto code = createKVCacheFile();
        if (code != NO_ERROR) {
            return code;
        }
        code = resetKVCacheFileSize(mCurrentKeySize, mCurrentValueSize);
        if (code != NO_ERROR) {
            return code;
        }
        code = expandKVCacheInDisk(oldKeySize, oldValueSize, mCurrentKeySize, mCurrentValueSize,
                                   (int)meta->seqlen_in_disk, true, (int)oldMaxLength, oldKeyFile.fd, oldValueFile.fd);
        if (code != NO_ERROR) {
            // Nothing of this allocation is usable: drop files, mappings and buffers
            // here instead of leaving a half-built cache for the next layer to trip on.
            onClear();
            mCurrentKeySize = 0;
            mCurrentValueSize = 0;
            return code;
        }

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
    size_t totalSize = (size_t)mBatch * mKvNumHead * mMaxLength * mHeadDim * keyByte;
    size_t valueTotalSize = (size_t)mBatch * mKvNumHead * mMaxLength * mHeadDim * valueByte;
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
        auto code = createKVCacheFile(keyStoredDst, valueStoredDst);
        if (code != NO_ERROR) {
            return code;
        }
        code = resetKVCacheFileSize(totalSize, valueTotalSize);
        if (code != NO_ERROR) {
            return code;
        }
        code = mmapKVCache(totalSize, valueTotalSize);
        if (code != NO_ERROR) {
            return code;
        }
        mKVCacheInDisk = true;
        auto abortAlloc = [&]() {
            onClear();
            mCurrentKeySize = 0;
            mCurrentValueSize = 0;
        };
        mKeyBuffer = [[context device] newBufferWithBytesNoCopy:mMapKeyAddr length:totalSize options:MTLResourceStorageModeShared  deallocator:nil];
        if (nil == mKeyBuffer) {
            MNN_ERROR("MNN::Metal: OUT_OF_MEMORY wrapping on-disk kv cache of %zu bytes\n", totalSize);
            abortAlloc();
            return OUT_OF_MEMORY;
        }
        mValueBuffer = [[context device] newBufferWithBytesNoCopy:mMapValueAddr length:valueTotalSize options:MTLResourceStorageModeShared  deallocator:nil];
        if (nil == mValueBuffer) {
            MNN_ERROR("MNN::Metal: OUT_OF_MEMORY wrapping on-disk kv cache of %zu bytes\n", valueTotalSize);
            abortAlloc();
            return OUT_OF_MEMORY;
        }
        if (useDynamicScaleBuffer()) {
            int scaleByte = mtbn->useFp16InsteadFp32() ? 2 : 4;
            const size_t scaleBytes = (size_t)mBatch * mMaxLength * scaleByte * 2;
            mKScaleBuffer = [[context device] newBufferWithLength:scaleBytes options:MTLResourceStorageModeShared];
            if (nil == mKScaleBuffer) {
                MNN_ERROR("MNN::Metal: OUT_OF_MEMORY allocating kv scale buffers for %d tokens\n", mMaxLength);
                abortAlloc();
                return OUT_OF_MEMORY;
            }
            mVScaleBuffer = [[context device] newBufferWithLength:scaleBytes options:MTLResourceStorageModeShared];
            if (nil == mVScaleBuffer) {
                MNN_ERROR("MNN::Metal: OUT_OF_MEMORY allocating kv scale buffers for %d tokens\n", mMaxLength);
                abortAlloc();
                return OUT_OF_MEMORY;
            }
            // Zero the whole scale/bias tail: dequant(0) = bias, so the K/V
            // memset above only makes padded tokens vanish if bias is 0 too.
            // Uninitialized bias bits can decode as Inf/NaN and 0*NaN = NaN
            // poisons the AV matmul for recycled (non-zero) pages.
            ::memset([mKScaleBuffer contents], 0, scaleBytes);
            ::memset([mVScaleBuffer contents], 0, scaleBytes);
        } else {
            mKScaleBuffer = nil;
            mVScaleBuffer = nil;
        }
        auto new_key_ptr = (uint8_t*)[mKeyBuffer contents];
        ::memset(new_key_ptr, 0, totalSize);
        auto new_value_ptr = (uint8_t*)[mValueBuffer contents];
        ::memset(new_value_ptr, 0, valueTotalSize);
    } else {
        // past_key: [maxlen, batch * kvNumhead, headdim]
        // past_value: [maxlen, batch * kvNumhead, headdim] (row-major, same as key)
        // The reservation is a target, not a requirement: fall back to the length this
        // forward needs rather than memset-ing a nil buffer.
        const int reservedLength = mMaxLength;
        const int neededLength = ROUND_UP(kv_seq_len + mConfig.mExpandChunk, mConfig.mKvAlignNum);
        std::unique_ptr<Tensor> new_key, new_value;
        auto acquireKV = [&](int length) {
            new_key.reset(Tensor::createDevice<int8_t>({length, mBatch * mKvNumHead, mHeadDim * keyByte}));
            new_value.reset(Tensor::createDevice<int8_t>({length, mBatch * mKvNumHead, mHeadDim * valueByte}));
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
        mCurrentKeySize = (size_t)mBatch * mKvNumHead * mMaxLength * mHeadDim * keyByte;
        mCurrentValueSize = (size_t)mBatch * mKvNumHead * mMaxLength * mHeadDim * valueByte;
        // memset for qkv matmul mad, in case dirty data
        auto newKeyBuf = MetalBackend::getBuffer(new_key.get());
        auto new_key_ptr = (uint8_t*)[newKeyBuf.first contents] + newKeyBuf.second;
        ::memset(new_key_ptr, 0, mCurrentKeySize);
        auto newValueBuf = MetalBackend::getBuffer(new_value.get());
        auto new_value_ptr = (uint8_t*)[newValueBuf.first contents] + newValueBuf.second;
        ::memset(new_value_ptr, 0, mCurrentValueSize);
        id<MTLBuffer> newKScale = nil, newVScale = nil;
        if (useDynamicScaleBuffer()) {
            int scaleByte = mtbn->useFp16InsteadFp32() ? 2 : 4;
            const size_t scaleBytes = (size_t)mBatch * mMaxLength * scaleByte * 2;
            newKScale = [[context device] newBufferWithLength:scaleBytes options:MTLResourceStorageModeShared];
            newVScale = [[context device] newBufferWithLength:scaleBytes options:MTLResourceStorageModeShared];
            if (nil == newKScale || nil == newVScale) {
                MNN_ERROR("MNN::Metal: OUT_OF_MEMORY allocating kv scale buffers for %d tokens, keep old cache\n", mMaxLength);
                // Only the K/V reservation was taken; the scale buffers were
                // never created, so ARC has nothing to release here.
                mBackend->onReleaseBuffer(new_key.get(), Backend::STATIC);
                mBackend->onReleaseBuffer(new_value.get(), Backend::STATIC);
                new_key.reset();
                new_value.reset();
                mMaxLength = prevMaxLength;
                mCurrentKeySize = prevKeySize;
                mCurrentValueSize = prevValueSize;
                return OUT_OF_MEMORY;
            }
            // See the in-disk branch above: zero scale+bias so padded tokens
            // dequantize to 0 instead of uninitialized (possibly NaN) bias.
            ::memset([newKScale contents], 0, scaleBytes);
            ::memset([newVScale contents], 0, scaleBytes);
        }
        mPastKey.reset(new_key.release());
        mPastValue.reset(new_value.release());
        mKScaleBuffer = newKScale;
        mVScaleBuffer = newVScale;

    }
    return NO_ERROR;
}
ErrorCode MetalKVCacheManager::growTo(int kvSeqLen, int pastLength) {
    auto mtbn = static_cast<MetalBackend *>(mBackend);
    int keyByte = mQuantKey ? 1 : (mtbn->useFp16InsteadFp32() ? 2 : 4);
    int valueByte = mQuantValue ? 1 : (mtbn->useFp16InsteadFp32() ? 2 : 4);
    // copy mPastLength including all remove/reverse to new buffer first
    auto copy_len = mPastLength;
    bool needCopy = mPastLength > 0;
    size_t old_size = (size_t)mBatch * mKvNumHead * copy_len * mHeadDim * keyByte;
    const auto oldKeyTotalSize = mCurrentKeySize;
    const auto oldValueTotalSize = mCurrentValueSize;
    const auto oldMaxLength = mMaxLength;
    const auto oldPastLength = mPastLength;
    // align max kv_seq_len to mKvAlignNum, for simd/tensor matrix load alignment
    mMaxLength = ROUND_UP(kvSeqLen + mConfig.mExpandChunk, mConfig.mKvAlignNum);
    mCurrentKeySize = (size_t)mBatch * mKvNumHead * mMaxLength * mHeadDim * keyByte;
    mCurrentValueSize = (size_t)mBatch * mKvNumHead * mMaxLength * mHeadDim * valueByte;
    mPastLength = pastLength;
    auto rollback = [&]() {
        mMaxLength = oldMaxLength;
        mCurrentKeySize = oldKeyTotalSize;
        mCurrentValueSize = oldValueTotalSize;
        mPastLength = oldPastLength;
    };
    if (mKVCacheInDisk) {
        auto code = expandKVCacheInDisk(oldKeyTotalSize, oldValueTotalSize, mCurrentKeySize, mCurrentValueSize,
                                        (int)copy_len, needCopy, oldMaxLength);
        if (code != NO_ERROR) {
            rollback();
            return code;
        }
        return NO_ERROR;
    }
    if (!expandKVCacheInMem(old_size, (int)copy_len, needCopy, oldMaxLength)) {
        rollback();
        return OUT_OF_MEMORY;
    }
    return NO_ERROR;
}

ErrorCode MetalKVCacheManager::ensureCapacity(int kvSeqLen) {
    if (kvSeqLen > mMaxLength) {
        invalidateReplay();
        return growTo(kvSeqLen, mPastLength);
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

        size_t old_size = (size_t)mBatch * mKvNumHead * copy_len * mHeadDim * keyByte;

        auto oldKeyTotalSize = mCurrentKeySize;
        auto oldValueTotalSize = mCurrentValueSize;
        auto oldMaxLength = mMaxLength;
        auto oldPastLength = mPastLength;
        // align max kv_seq_len to mKvAlignNum, for simd/tensor matrix load alignment
        capacity = std::max(capacity, kv_seq_len + mConfig.mExpandChunk);
        mMaxLength = ROUND_UP(capacity, mConfig.mKvAlignNum);

        mCurrentKeySize = (size_t)mBatch * mKvNumHead * mMaxLength * mHeadDim * keyByte;
        mCurrentValueSize = (size_t)mBatch * mKvNumHead * mMaxLength * mHeadDim * valueByte;

        mPastLength = (int)start;

        if(mKVCacheInDisk) {
            auto code = expandKVCacheInDisk(oldKeyTotalSize, oldValueTotalSize, mCurrentKeySize, mCurrentValueSize,
                                            (int)copy_len, needCopy, oldMaxLength);
            if (code != NO_ERROR) {
                // The old mapping is already gone; compacting into it would dereference null.
                mMaxLength = oldMaxLength;
                mCurrentKeySize = oldKeyTotalSize;
                mCurrentValueSize = oldValueTotalSize;
                mPastLength = copy_len;
                return code;
            }
        } else if (!expandKVCacheInMem(old_size, (int)copy_len, needCopy, oldMaxLength)) {
            // A failed reservation must not leave the cache smaller than the tokens this
            // forward writes, so retry at the organic need before keeping the old cache.
            const int neededLength = ROUND_UP(kv_seq_len + mConfig.mExpandChunk, mConfig.mKvAlignNum);
            bool retryOk = false;
            if (neededLength < mMaxLength) {
                mMaxLength = neededLength;
                mCurrentKeySize = (size_t)mBatch * mKvNumHead * mMaxLength * mHeadDim * keyByte;
                mCurrentValueSize = (size_t)mBatch * mKvNumHead * mMaxLength * mHeadDim * valueByte;
                retryOk = expandKVCacheInMem(old_size, (int)copy_len, needCopy, oldMaxLength);
            }
            if (!retryOk) {
                // Nothing was expanded: the cache still holds exactly the old tokens.
                mMaxLength = oldMaxLength;
                mCurrentKeySize = oldKeyTotalSize;
                mCurrentValueSize = oldValueTotalSize;
                mPastLength = oldPastLength;
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
            if (key_ptr == nullptr || value_ptr == nullptr) {
                return OUT_OF_MEMORY;
            }
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
            const size_t keyStep = (size_t)mBatch * mKvNumHead * mHeadDim * keyByte;
            const size_t valueStep = (size_t)mBatch * mKvNumHead * mHeadDim * valueByte;
            for(int i = 0; i < length; i++) {
                ::memcpy(key_ptr + (copy_dst_index + i) * keyStep, key_ptr + (copy_src_index + i) * keyStep, keyStep);
            }
            for(int i = 0; i < length; i++) {
                ::memcpy(value_ptr + (copy_dst_index + i) * valueStep, value_ptr + (copy_src_index + i) * valueStep, valueStep);
            }
            if (mKScaleBuffer != nil) {
                int scaleByte = mtbn->useFp16InsteadFp32() ? 2 : 4;
                auto* k_scale_bytes = (uint8_t*)[mKScaleBuffer contents];
                auto* v_scale_bytes = (uint8_t*)[mVScaleBuffer contents];
                const size_t kStep = (size_t)mBatch * 2 * scaleByte;
                const size_t vPair = (size_t)2 * scaleByte;
                for(int i = 0; i < length; i++) {
                    ::memcpy(k_scale_bytes + (copy_dst_index + i) * kStep,
                             k_scale_bytes + (copy_src_index + i) * kStep, kStep);
                    for (int b = 0; b < mBatch; ++b) {
                        const size_t batchBase = (size_t)b * mMaxLength;
                        ::memcpy(v_scale_bytes + (batchBase + copy_dst_index + i) * vPair,
                                 v_scale_bytes + (batchBase + copy_src_index + i) * vPair, vPair);
                    }
                }
            }
            start += length;
        }
        mPastLength = (int)start;
    }
    return NO_ERROR;
}

bool MetalKVCacheManager::expandKVCacheInMem(size_t oldSize, int copy_len, bool need_copy, int oldMaxLength) {
    auto mtbn = static_cast<MetalBackend *>(mBackend);
    int keyByte = mQuantKey ? 1 : (mtbn->useFp16InsteadFp32() ? 2 : 4);
    int valueByte = mQuantValue ? 1 : (mtbn->useFp16InsteadFp32() ? 2 : 4);
    // past_key: [maxlen, batch * kvNumhead, headdim]
    std::unique_ptr<Tensor> new_key(Tensor::createDevice<int8_t>({mMaxLength, mBatch * mKvNumHead, mHeadDim * keyByte}));
    // past_value: [maxlen, batch * kvNumhead, headdim] (row-major, same as key)
    std::unique_ptr<Tensor> new_value(Tensor::createDevice<int8_t>({mMaxLength, mBatch * mKvNumHead, mHeadDim * valueByte}));

    const bool keyAcquired = mBackend->onAcquireBuffer(new_key.get(), Backend::STATIC);
    const bool valueAcquired = keyAcquired && mBackend->onAcquireBuffer(new_value.get(), Backend::STATIC);
    auto releaseAcquired = [&]() {
        if (valueAcquired) {
            mBackend->onReleaseBuffer(new_value.get(), Backend::STATIC);
        }
        if (keyAcquired) {
            mBackend->onReleaseBuffer(new_key.get(), Backend::STATIC);
        }
    };
    if (!valueAcquired) {
        MNN_ERROR("MNN::Metal: OUT_OF_MEMORY expanding kv cache to %d, keep old cache\n", mMaxLength);
        releaseAcquired();
        return false;
    }
    auto newKeyBuf = MetalBackend::getBuffer(new_key.get());
    auto newValueBuf = MetalBackend::getBuffer(new_value.get());
    if (nil == newKeyBuf.first || nil == newValueBuf.first) {
        MNN_ERROR("MNN::Metal: OUT_OF_MEMORY expanding kv cache to %d, keep old cache\n", mMaxLength);
        releaseAcquired();
        return false;
    }

    // memset for qkv matmul mad, in case dirty data
    auto new_key_ptr = (uint8_t*)[newKeyBuf.first contents] + newKeyBuf.second;
    ::memset(new_key_ptr, 0, (size_t)mBatch * mMaxLength * mKvNumHead * mHeadDim * keyByte);

    auto new_value_ptr = (uint8_t*)[newValueBuf.first contents] + newValueBuf.second;
    ::memset(new_value_ptr, 0, (size_t)mBatch * mMaxLength * mKvNumHead * mHeadDim * valueByte);

    if (need_copy) {
        auto keyBuf = MetalBackend::getBuffer(mPastKey.get());
        auto key_ptr = (uint8_t*)[keyBuf.first contents] + keyBuf.second;
        ::memcpy(new_key_ptr, key_ptr, oldSize);

        auto valueBuf = MetalBackend::getBuffer(mPastValue.get());
        auto value_ptr = (uint8_t*)[valueBuf.first contents] + valueBuf.second;
        ::memcpy(new_value_ptr, value_ptr, (size_t)mBatch * copy_len * mKvNumHead * mHeadDim * valueByte);
    }

    auto context = (__bridge MNNMetalContext *)mtbn->context();
    id<MTLBuffer> newKScale = nil, newVScale = nil;
    if (useDynamicScaleBuffer()) {
        int scaleByte = mtbn->useFp16InsteadFp32() ? 2 : 4;
        const size_t scaleBytes = (size_t)mBatch * mMaxLength * scaleByte * 2;
        newKScale = [[context device] newBufferWithLength:scaleBytes options:MTLResourceStorageModeShared];
        newVScale = [[context device] newBufferWithLength:scaleBytes options:MTLResourceStorageModeShared];
        if (nil == newKScale || nil == newVScale) {
            MNN_ERROR("MNN::Metal: OUT_OF_MEMORY expanding kv scale to %d, keep old cache\n", mMaxLength);
            releaseAcquired();
            return false;
        }
    }

    mPastKey.reset(new_key.release());
    mPastValue.reset(new_value.release());

    if (useDynamicScaleBuffer()) {
        int scaleByte = mtbn->useFp16InsteadFp32() ? 2 : 4;
        const size_t scaleBytes = (size_t)mBatch * mMaxLength * scaleByte * 2;
        // Zero before the copy so the grown tail past copy_len is not left
        // uninitialized (padded tokens must dequantize to 0, see onAlloc).
        ::memset([newKScale contents], 0, scaleBytes);
        ::memset([newVScale contents], 0, scaleBytes);
        if (need_copy && mKScaleBuffer != nil) {
            ::memcpy([newKScale contents], [mKScaleBuffer contents], (size_t)mBatch * copy_len * scaleByte * 2);
            for (int b = 0; b < mBatch; ++b) {
                ::memcpy((uint8_t*)[newVScale contents] + (size_t)b * mMaxLength * scaleByte * 2,
                         (const uint8_t*)[mVScaleBuffer contents] + (size_t)b * oldMaxLength * scaleByte * 2,
                         (size_t)copy_len * scaleByte * 2);
            }
        }
        mKScaleBuffer = newKScale;
        mVScaleBuffer = newVScale;
    } else {
        mKScaleBuffer = nil;
        mVScaleBuffer = nil;
    }
    return true;
}

ErrorCode MetalKVCacheManager::expandKVCacheInDisk(size_t oldKeySize, size_t oldValueSize, size_t curKeySize,
                                                   size_t curValueSize, int copy_len, bool need_copy,
                                                   int oldMaxLength, file_t specKeyFile, file_t specValueFile) {
    auto mtbn = static_cast<MetalBackend *>(mBackend);
    auto context = (__bridge MNNMetalContext *)mtbn->context();

    auto code = mmapKVCache(oldKeySize, oldValueSize, specKeyFile, specValueFile);
    if (code != NO_ERROR) {
        mKeyBuffer = nil;
        mValueBuffer = nil;
        return code;
    }
    std::unique_ptr<int8_t[]> prevKey, prevValue;
    if (need_copy) {
        prevKey.reset(new (std::nothrow) int8_t[oldKeySize]);
        prevValue.reset(new (std::nothrow) int8_t[oldValueSize]);
        if (nullptr == prevKey.get() || nullptr == prevValue.get()) {
            MNN_ERROR("MNN::Metal: OUT_OF_MEMORY backing up kv cache of %zu bytes\n", oldKeySize);
            unmapKVCache(oldKeySize, oldValueSize);
            mKeyBuffer = nil;
            mValueBuffer = nil;
            return OUT_OF_MEMORY;
        }
        ::memcpy(prevKey.get(),   mMapKeyAddr,   oldKeySize);
        ::memcpy(prevValue.get(), mMapValueAddr, oldValueSize);
    }

    unmapKVCache(oldKeySize, oldValueSize);
    code = resetKVCacheFileSize(curKeySize, curValueSize);
    if (code != NO_ERROR) {
        mKeyBuffer = nil;
        mValueBuffer = nil;
        return code;
    }
    code = mmapKVCache(curKeySize, curValueSize);
    if (code != NO_ERROR) {
        // The mapping helper already unmapped the half pair; drop the wrappers
        // here so no caller can encode against a stale MTLBuffer.
        mKeyBuffer = nil;
        mValueBuffer = nil;
        return code;
    }

    // reset id<MTLBuffer>: build every replacement first and only publish the
    // ones that exist; a nil buffer must never be handed to the shaders.
    auto newKeyBuffer = [[context device] newBufferWithBytesNoCopy:mMapKeyAddr length:curKeySize options:MTLResourceStorageModeShared  deallocator:nil];
    if (nil == newKeyBuffer) {
        MNN_ERROR("MNN::Metal: OUT_OF_MEMORY expanding on-disk kv cache to %zu bytes\n", curKeySize);
        unmapKVCache(curKeySize, curValueSize);
        mKeyBuffer = nil;
        mValueBuffer = nil;
        return OUT_OF_MEMORY;
    }
    auto newValueBuffer = [[context device] newBufferWithBytesNoCopy:mMapValueAddr length:curValueSize options:MTLResourceStorageModeShared  deallocator:nil];
    if (nil == newValueBuffer) {
        MNN_ERROR("MNN::Metal: OUT_OF_MEMORY expanding on-disk kv cache to %zu bytes\n", curValueSize);
        unmapKVCache(curKeySize, curValueSize);
        mKeyBuffer = nil;
        mValueBuffer = nil;
        return OUT_OF_MEMORY;
    }
    id<MTLBuffer> newKScale = nil, newVScale = nil;
    if (useDynamicScaleBuffer()) {
        int scaleByte = mtbn->useFp16InsteadFp32() ? 2 : 4;
        const size_t scaleBytes = (size_t)mBatch * mMaxLength * scaleByte * 2;
        newKScale = [[context device] newBufferWithLength:scaleBytes options:MTLResourceStorageModeShared];
        newVScale = [[context device] newBufferWithLength:scaleBytes options:MTLResourceStorageModeShared];
        if (nil == newKScale || nil == newVScale) {
            MNN_ERROR("MNN::Metal: OUT_OF_MEMORY expanding on-disk kv scale to %d tokens\n", mMaxLength);
            unmapKVCache(curKeySize, curValueSize);
            mKeyBuffer = nil;
            mValueBuffer = nil;
            return OUT_OF_MEMORY;
        }
    }
    mKeyBuffer   = newKeyBuffer;
    mValueBuffer = newValueBuffer;

    if (useDynamicScaleBuffer()) {
        int scaleByte = mtbn->useFp16InsteadFp32() ? 2 : 4;
        const size_t scaleBytes = (size_t)mBatch * mMaxLength * scaleByte * 2;
        // Zero before the copy so the grown tail past copy_len is not left
        // uninitialized (padded tokens must dequantize to 0, see onAlloc).
        ::memset([newKScale contents], 0, scaleBytes);
        ::memset([newVScale contents], 0, scaleBytes);
        if (need_copy && mKScaleBuffer != nil) {
            ::memcpy([newKScale contents], [mKScaleBuffer contents], (size_t)mBatch * copy_len * scaleByte * 2);
            for (int b = 0; b < mBatch; ++b) {
                ::memcpy((uint8_t*)[newVScale contents] + (size_t)b * mMaxLength * scaleByte * 2,
                         (const uint8_t*)[mVScaleBuffer contents] + (size_t)b * oldMaxLength * scaleByte * 2,
                         (size_t)copy_len * scaleByte * 2);
            }
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
        ::memcpy(mMapKeyAddr, prevKey.get(), oldKeySize);
        ::memcpy(mMapValueAddr, prevValue.get(), oldValueSize);
    }
    return NO_ERROR;
}

void MetalKVCacheManager::onClear() {
    invalidateReplay();
    if (mKVCacheInDisk) {
        mKeyBuffer = nil;
        mValueBuffer = nil;

        unmapKVCache(mCurrentKeySize, mCurrentValueSize);
        mKVCacheInDisk = false;
    }
    // A retained prefix owns its files, but the descriptors are ours either way;
    // a failed alloc also leaves files behind without mKVCacheInDisk ever set.
    if (mSaveShareKvPrefix) {
        closeKVCacheFile();
    } else {
        removeKVCacheFile();
    }
    mSaveShareKvPrefix = false;
    mPastKey.reset();
    mPastValue.reset();
    mKScaleBuffer = nil;
    mVScaleBuffer = nil;
    mMaxLength = 0;
    mPastLength = 0;
}
} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
