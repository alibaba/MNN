//
//  SoftmaxBufExecution.cpp
//  MNN
//
//  Created by MNN on 2024/04/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "backend/opencl/execution/buffer/AttentionBufExecution.hpp"
#include "core/MNNFileUtils.h"
#include <fstream>
namespace MNN {
namespace OpenCL {

// Which prefill implementation onResize picks. Persisted in the tune cache, so the
// values are part of the on-disk format: only append. 0 was short prefill, deleted --
// an older cache can still name it, which is why the mode read back gets validated.
enum PrefillMode { kPrefillLong = 1, kPrefillFlash = 2 };

// Measured winners for the kv-cache rearrange kernels; handing them to the tuner keeps Wide from
// sweeping the full grid on every kv-cache resize.
static LwsShortlist rearrangeKLwsShortlist(GpuType gpuType) {
    static const uint32_t adrenoPool[][3] = {{1, 8, 4}, {2, 4, 4}, {2, 8, 4}, {4, 8, 4}};
    static const uint32_t maliPool[][3] = {{1, 2, 2}, {1, 32, 2}};
    return makeLwsShortlist(gpuType, adrenoPool, maliPool);
}

static LwsShortlist rearrangeVLwsShortlist(GpuType gpuType) {
    static const uint32_t adrenoPool[][3] = {{8, 2, 1},  {8, 4, 1},  {8, 1, 2},  {4, 8, 1}, {2, 16, 1},
                                             {16, 2, 1}, {16, 4, 1}, {32, 1, 1}, {2, 32, 1}};
    static const uint32_t maliPool[][3] = {{1, 4, 1}, {2, 2, 1}, {1, 8, 1}};
    return makeLwsShortlist(gpuType, adrenoPool, maliPool);
}

KVCacheCLManager::KVCacheCLManager(Backend* backend, bool kv_cahce) : mKVCache(kv_cahce) {
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
}

void KVCacheCLManager::allocKVCache(const KVMeta* meta, int seqlen) {
    if (!mKVCache) {
        return;
    }
    if (mOpenCLBackend->getPrecision() != BackendConfig::Precision_High) {
        mByte = 2;
    }

    // Prefix kvcache: share a fixed prompt's kvcache on disk (see Llm::setPrefixCacheFile).
    // Only act on the very first allocation of this layer's cache: file_flag stays
    // PendingWrite/PendingRead through the decode steps of the same generate() call, but the
    // load/save + layer_index bookkeeping must happen exactly once (mirrors onAlloc-once on
    // CPU/Metal, where growth is handled by onRealloc without prefix logic).
    bool firstAlloc = (mPastKey.get() == nullptr);
    bool hasPrefixFile = firstAlloc && meta != nullptr && meta->file_name.size() > 0 && !mPrefixCacheDir.empty();

    // Load path: pull the per-layer prefix files into the cache and skip prefill compute.
    if (hasPrefixFile && meta->file_flag == KVMeta::PendingRead) {
        if (loadPrefixKVCache(meta, seqlen)) {
            mReallocDone = true;
            return;
        }
        // Fall through to a normal (empty) allocation when the cache is unusable.
    }

    // Save path: remember this layer's target file, then allocate normally. onExecute
    // dumps the prefill kvcache to disk once the kernels have run.
    if (hasPrefixFile && meta->file_flag == KVMeta::PendingWrite) {
        mSaveShareKvPrefix = true;
        if (!MNNCreateDir(mPrefixCacheDir.c_str())) {
            MNN_PRINT("Failed to create prefix cache file dir: %s\n", mPrefixCacheDir.c_str());
        }
        mBasePrefixFileName =
            MNNFilePathConcat(mPrefixCacheDir, meta->file_name) + "_" + std::to_string(meta->layer_index);
        // Advance the shared per-layer counter exactly once during the resize pass,
        // matching CPUKVCacheManager / MetalKVCacheManager.
        const_cast<KVMeta*>(meta)->layer_index = (meta->layer_index + 1) % meta->layer_nums;
    }

    mPastLength = meta != nullptr ? meta->previous : 0;
    // Fully execute reallocKVCache (including Remove and mPastLength update)
    // in resize phase so that LWS tuning kernels use the same args as execute.
    reallocKVCache(meta, seqlen, true);
    mReallocDone = true;
}

bool KVCacheCLManager::loadPrefixKVCache(const KVMeta* meta, int seqlen) {
    std::string pathk =
        MNNFilePathConcat(mPrefixCacheDir, meta->file_name) + "_" + std::to_string(meta->layer_index) + ".k";
    std::string pathv =
        MNNFilePathConcat(mPrefixCacheDir, meta->file_name) + "_" + std::to_string(meta->layer_index) + ".v";
    // Advance the shared per-layer counter once during the resize pass.
    const_cast<KVMeta*>(meta)->layer_index = (meta->layer_index + 1) % meta->layer_nums;

    int diskLen = meta->seqlen_in_disk;
    if (diskLen <= 0) {
        MNN_PRINT("Prefix cache: invalid seqlen_in_disk %d\n", diskLen);
        return false;
    }

    // Compact on-disk layout (stride == diskLen, no 4-alignment padding):
    //   key   : [kvNumHead * headDim, diskLen]
    //   value : [kvNumHead, diskLen, headDim]
    size_t expectKeyBytes = (size_t)mKvNumHead * mHeadDim * diskLen * mByte;
    size_t expectValueBytes = (size_t)mKvNumHead * diskLen * mHeadDim * mByte;

    auto keyFd = MNNOpenFile(pathk.c_str(), MNN_FILE_READ);
    auto valueFd = MNNOpenFile(pathv.c_str(), MNN_FILE_READ);
    if (keyFd == INVALID_FILE || valueFd == INVALID_FILE) {
        MNN_PRINT("Prefix cache: failed to open %s / %s\n", pathk.c_str(), pathv.c_str());
        if (keyFd != INVALID_FILE)
            MNNCloseFile(keyFd);
        if (valueFd != INVALID_FILE)
            MNNCloseFile(valueFd);
        return false;
    }
    size_t keyBytes = MNNGetFileSize(keyFd);
    size_t valueBytes = MNNGetFileSize(valueFd);
    // The file must be at least as large as the region we will read. A mismatch usually
    // means the cache was written with a different precision/shape; bail to a clean cache.
    if (keyBytes < expectKeyBytes || valueBytes < expectValueBytes) {
        MNN_PRINT("Prefix cache size mismatch: key %zu(expect>=%zu) value %zu(expect>=%zu), rebuild cache\n", keyBytes,
                  expectKeyBytes, valueBytes, expectValueBytes);
        MNNCloseFile(keyFd);
        MNNCloseFile(valueFd);
        return false;
    }

    // Match the buffer size the no-cache path would produce so the attention kernels use the
    // same maxLength (hence the same fp16 tiling/reduction order) and the output is bit-identical:
    // the no-cache path allocs diskLen+mExpandChunk for the prefix, then only grows if the query
    // pushes past it (seqlen > mExpandChunk).
    int kvNeed = diskLen + seqlen;
    mMaxLength = (kvNeed <= diskLen + mExpandChunk) ? (diskLen + mExpandChunk) : (kvNeed + mExpandChunk);
    size_t newMaxlen = ROUND_UP(mMaxLength, 4);
    size_t bufferSize = UP_DIV(mMaxLength, 4) * mKvNumHead * mHeadDim * 4 * mByte;
    cl_int res;
    auto newKey = new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(),
                                 CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize);
    auto newValue = new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(),
                                   CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize);

    // Load key: [kvNumHead * headDim, diskLen] -> buffer [kvNumHead * headDim, newMaxlen]
    {
        char* keyPtr = (char*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
            *newKey, true, CL_MAP_WRITE, 0, bufferSize, nullptr, nullptr, &res);
        if (keyPtr != nullptr && res == CL_SUCCESS) {
            ::memset(keyPtr, 0, bufferSize);
            size_t diskRow = (size_t)diskLen * mByte;
            size_t bufRow = newMaxlen * mByte;
            for (int i = 0; i < mKvNumHead * mHeadDim; ++i) {
                MNNSetFilePointer(keyFd, (size_t)i * diskRow);
                MNNReadFile(keyFd, keyPtr + i * bufRow, diskRow);
            }
        } else {
            MNN_ERROR("Prefix cache: map new key failed\n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*newKey, keyPtr);
    }

    // Load value: [kvNumHead, diskLen, headDim] -> buffer [kvNumHead, newMaxlen, headDim]
    {
        char* valuePtr = (char*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
            *newValue, true, CL_MAP_WRITE, 0, bufferSize, nullptr, nullptr, &res);
        if (valuePtr != nullptr && res == CL_SUCCESS) {
            ::memset(valuePtr, 0, bufferSize);
            size_t rowBytes = (size_t)mHeadDim * mByte;
            for (int h = 0; h < mKvNumHead; ++h) {
                MNNSetFilePointer(valueFd, (size_t)h * diskLen * rowBytes);
                // Contiguous [diskLen, headDim] block on disk maps to [newMaxlen, headDim]
                // buffer rows; per-head strides differ so read the whole block at once.
                MNNReadFile(valueFd, valuePtr + (size_t)h * newMaxlen * rowBytes, (size_t)diskLen * rowBytes);
            }
        } else {
            MNN_ERROR("Prefix cache: map new value failed\n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*newValue, valuePtr);
    }

    MNNCloseFile(keyFd);
    MNNCloseFile(valueFd);
    mPastKey.reset(newKey);
    mPastValue.reset(newValue);
    mPastLength = diskLen;
    return true;
}

void KVCacheCLManager::savePrefixKVCache() {
    if (!mSaveShareKvPrefix || mBasePrefixFileName.empty() || mPastLength <= 0) {
        return;
    }
    if (mPastKey.get() == nullptr || mPastValue.get() == nullptr) {
        return;
    }
    // Ensure all kvcache writes issued by the attention kernels are complete before mapping.
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().finish();

    int len = mPastLength;
    size_t newMaxlen = ROUND_UP(mMaxLength, 4);
    size_t bufferSize = UP_DIV(mMaxLength, 4) * mKvNumHead * mHeadDim * 4 * mByte;
    cl_int res;

    std::string pathk = mBasePrefixFileName + ".k";
    std::string pathv = mBasePrefixFileName + ".v";
    auto keyFd = MNNCreateFile(pathk.c_str());
    auto valueFd = MNNCreateFile(pathv.c_str());
    if (keyFd == INVALID_FILE || valueFd == INVALID_FILE) {
        MNN_PRINT("Prefix cache: failed to create %s / %s\n", pathk.c_str(), pathv.c_str());
        if (keyFd != INVALID_FILE)
            MNNCloseFile(keyFd);
        if (valueFd != INVALID_FILE)
            MNNCloseFile(valueFd);
        return;
    }

    // Dump key: buffer [kvNumHead * headDim, newMaxlen] -> compact [kvNumHead * headDim, len]
    {
        char* keyPtr = (char*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
            *mPastKey.get(), true, CL_MAP_READ, 0, bufferSize, nullptr, nullptr, &res);
        if (keyPtr != nullptr && res == CL_SUCCESS) {
            size_t diskRow = (size_t)len * mByte;
            size_t bufRow = newMaxlen * mByte;
            for (int i = 0; i < mKvNumHead * mHeadDim; ++i) {
                MNNWriteFile(keyFd, keyPtr + i * bufRow, diskRow);
            }
        } else {
            MNN_ERROR("Prefix cache: map key for save failed\n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mPastKey.get(), keyPtr);
    }

    // Dump value: buffer [kvNumHead, newMaxlen, headDim] -> compact [kvNumHead, len, headDim]
    {
        char* valuePtr = (char*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
            *mPastValue.get(), true, CL_MAP_READ, 0, bufferSize, nullptr, nullptr, &res);
        if (valuePtr != nullptr && res == CL_SUCCESS) {
            size_t rowBytes = (size_t)mHeadDim * mByte;
            for (int h = 0; h < mKvNumHead; ++h) {
                MNNWriteFile(valueFd, valuePtr + (size_t)h * newMaxlen * rowBytes, (size_t)len * rowBytes);
            }
        } else {
            MNN_ERROR("Prefix cache: map value for save failed\n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mPastValue.get(), valuePtr);
    }

    MNNCloseFile(keyFd);
    MNNCloseFile(valueFd);
    // Only dump once per prefill; _sync markers are created by Llm::completePrefixWrite.
    mSaveShareKvPrefix = false;
}

bool KVCacheCLManager::reallocKVCache(const KVMeta* meta, int seqlen, bool isExecute) {
    if (!mKVCache) {
        return false;
    }
    // Sync internal KV length to framework's authoritative value; prevents runaway accumulation
    // when onResize is skipped across repeated same-shape forwards (e.g. DiT diffusion steps).
    mPastLength = meta->previous;
    int kvSeqlen = meta->previous + seqlen - meta->remove + meta->computeReverseSize();
    int start = mPastLength - meta->remove;
    cl_int res;

    // latest length larger than maxLen
    // Also check if the 4-aligned write range exceeds the buffer's aligned capacity.
    // Kernels write in groups of 4 along seq_len, so past_len + ROUND_UP(seqlen, 4)
    // must not exceed ROUND_UP(mMaxLength, 4) to avoid out-of-bounds GPU memory access.
    int pastLen = meta->previous - meta->remove + meta->computeReverseSize();
    int alignedWriteEnd = pastLen + ROUND_UP(seqlen, 4);
    if (kvSeqlen > mMaxLength || alignedWriteEnd > (int)ROUND_UP(mMaxLength, 4)) {
        int copylen = mPastLength - meta->remove + meta->computeReverseSize();
        // Guard: skip copy when old KV buffers don't exist yet (first allocation).
        bool needCopy = copylen > 0 && mPastKey.get() != nullptr && mPastValue.get() != nullptr;

        size_t oldSize = mKvNumHead * UP_DIV(mMaxLength, 4) * mHeadDim * 4 * mByte;
        size_t oldMaxlen = ROUND_UP(mMaxLength, 4);
        mMaxLength = kvSeqlen + mExpandChunk;
        size_t newMaxlen = ROUND_UP(mMaxLength, 4);
        size_t bufferSize = UP_DIV(mMaxLength, 4) * mKvNumHead * mHeadDim * 4 * mByte;
        // past_key: [1, numhead, headdim, maxlen]
        auto newKey = new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(),
                                     CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize);
        // past_value: [1, numhead, maxlen, headdim]
        auto newValue = new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(),
                                       CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize);

        // Zero the freshly allocated buffers (matching CPU/Metal and loadPrefixKVCache). The
        // attention kernels read the 4-aligned tail [kvSeqlen, ROUND_UP(kvSeqlen, 4)); leaving
        // it uninitialized makes the result depend on stale device memory and diverge from the
        // disk-loaded prefix path, which zeroes its padding.
        for (cl::Buffer* buf : {newKey, newValue}) {
            char* p = (char*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
                *buf, true, CL_MAP_WRITE, 0, bufferSize, nullptr, nullptr, &res);
            if (p != nullptr && res == CL_SUCCESS) {
                ::memset(p, 0, bufferSize);
                mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*buf, p);
            }
        }

        if (needCopy) {
            // copy key
            {
                size_t oldMaxlenSize = oldMaxlen * mByte;
                size_t newMaxlenSize = newMaxlen * mByte;
                char* newKeyPtr = (char*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
                    *newKey, true, CL_MAP_WRITE, 0, bufferSize, nullptr, nullptr, &res);
                char* keyPtr = (char*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
                    *mPastKey.get(), true, CL_MAP_READ, 0, oldSize, nullptr, nullptr, &res);
                if (newKeyPtr != nullptr && keyPtr != nullptr && res == CL_SUCCESS) {
                    for (int i = 0; i < mKvNumHead * mHeadDim; ++i) {
                        ::memcpy(newKeyPtr + i * newMaxlenSize, keyPtr + i * oldMaxlenSize, oldMaxlenSize);
                    }
                } else {
                    MNN_ERROR("Map error key_ptr == nullptr \n");
                    MNN_ASSERT(false);
                }
                mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*newKey, newKeyPtr);
                mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mPastKey.get(), keyPtr);
            }

            // copy value
            {
                char* newValuePtr = (char*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
                    *newValue, true, CL_MAP_WRITE, 0, bufferSize, nullptr, nullptr, &res);
                char* valuePtr = (char*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
                    *mPastValue.get(), true, CL_MAP_READ, 0, oldSize, nullptr, nullptr, &res);
                if (newValuePtr != nullptr && valuePtr != nullptr && res == CL_SUCCESS) {
                    for (int i = 0; i < mKvNumHead; ++i) {
                        for (int j = 0; j < copylen; ++j) {
                            ::memcpy(newValuePtr + (i * newMaxlen + j) * mHeadDim * mByte,
                                     valuePtr + (i * oldMaxlen + j) * mHeadDim * mByte, mHeadDim * mByte);
                        }
                    }
                } else {
                    MNN_ERROR("Map error value_ptr == nullptr \n");
                    MNN_ASSERT(false);
                }
                mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*newValue, newValuePtr);
                mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mPastValue.get(), valuePtr);
            }
        }
        mPastKey.reset(newKey);
        mPastValue.reset(newValue);
        // resize phase don't update mPastLength value, excute phase will update it
        if (isExecute) {
            mPastLength = start;
        }
    }

    // Remove
    // resize phase don't remove kvcache, excute phase will do it
    if (isExecute) {
        if (0 == meta->n_reserve) {
            mPastLength = start;
            return true;
        }
        size_t curMaxlen = ROUND_UP(mMaxLength, 4);
        size_t pastkvSize = mKvNumHead * UP_DIV(curMaxlen, 4) * mHeadDim * 4 * mByte;
        char* keyPtr = (char*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
            *mPastKey.get(), true, CL_MAP_READ | CL_MAP_WRITE, 0, pastkvSize, nullptr, nullptr, &res);
        char* valuePtr = (char*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
            *mPastValue.get(), true, CL_MAP_READ | CL_MAP_WRITE, 0, pastkvSize, nullptr, nullptr, &res);

        // TODO: need to ensure reserve info is sorted
        auto copyDstIndex = start;
        for (int n = 0; n < meta->n_reserve; ++n) {
            auto begin = meta->reserve[2 * n];
            auto length = meta->reserve[2 * n + 1];
            // past_key   : [mKvNumHead, mHeadDim, mMaxLength]
            // past_value : [mKvNumHead, mMaxLength, mHeadDim]

            auto copySrcIndex = start + begin;
            for (int i = 0; i < mKvNumHead * mHeadDim; i++) {
                ::memmove(keyPtr + (i * curMaxlen + copyDstIndex) * mByte,
                          keyPtr + (i * curMaxlen + copySrcIndex) * mByte, length * mByte);
            }
            for (int i = 0; i < mKvNumHead; i++) {
                for (int j = 0; j < length; j++) {
                    ::memmove(valuePtr + (i * curMaxlen + copyDstIndex + j) * mHeadDim * mByte,
                              valuePtr + (i * curMaxlen + copySrcIndex + j) * mHeadDim * mByte, mHeadDim * mByte);
                }
            }
            copyDstIndex += length;
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mPastKey.get(), keyPtr);
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mPastValue.get(), valuePtr);
        mPastLength = (int)copyDstIndex;
    }
    return true;
}

void AttentionBufExecution::handleKVCache(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto query = inputs[0];
    auto key = inputs[1];
    auto shape = query->shape();

    int batch = shape[0];
    int seqlen = shape[1];
    int kvInputLen = key->shape()[1];
    int numHead = shape[2];
    int kvNumHead = key->shape()[2];
    int headDim = shape[3];

    if (nullptr == mMeta) {
        mPastKvSeqlen = 0;
        mKvSeqlen = kvInputLen;
        mKeyValueMaxlen = ROUND_UP(kvInputLen, 4);
        mDecodeTmpMaxlen = ROUND_UP(kvInputLen, 4);
        return;
    }
    mKVCacheCLManager->setArgs(numHead, kvNumHead, headDim);
    mKVCacheCLManager->allocKVCache(mMeta, kvInputLen);
    mKeyValueMaxlen = ROUND_UP(mKVCacheCLManager->maxLength(), 4);
    mDecodeTmpMaxlen = mKeyValueMaxlen;
    mPastKvSeqlen = mKVCacheCLManager->pastKvLength();
    mKvSeqlen = mPastKvSeqlen + kvInputLen;
}

ErrorCode AttentionBufExecution::init() {
    if (nullptr == mMeta) {
        return NO_ERROR;
    }
    // clear update arg vector, if prefill and decode use the same one
    mOpRecordUpdateInfo.clear();
    mRgUpdateInfo.update_kernel_args.clear();
    mRgUpdateInfo.update_global_size.clear();
    mRgUpdateInfo.update_local_size.clear();
    mRgVUpdateInfo.update_kernel_args.clear();
    mRgVUpdateInfo.update_global_size.clear();
    mRgVUpdateInfo.update_local_size.clear();
    mFaUpdateInfo.update_kernel_args.clear();
    mFaUpdateInfo.update_global_size.clear();
    mFaUpdateInfo.update_local_size.clear();
    mFdPartialUpdateInfo.update_kernel_args.clear();
    mFdPartialUpdateInfo.update_global_size.clear();
    mFdPartialUpdateInfo.update_local_size.clear();
    mFdReduceUpdateInfo.update_kernel_args.clear();
    mFdReduceUpdateInfo.update_global_size.clear();
    mFdReduceUpdateInfo.update_local_size.clear();

    return NO_ERROR;
}

ErrorCode AttentionBufExecution::UpdateArgs(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (nullptr == mMeta) {
        return NO_ERROR;
    }

    auto query = inputs[0];
    auto key = inputs[1];
    auto shape = query->shape();

    int kvInputLen = key->shape()[1];
    int numHead = shape[2];
    int headDim = shape[3];
    mPastKvSeqlen = mKVCacheCLManager->pastKvLength();
    mKvSeqlen = mKVCacheCLManager->pastKvLength() + kvInputLen;
    mKVCacheCLManager->addKvLength(kvInputLen);
    // prefill
    if (mIsDecode == false) {
        // key value static memory has been changed, need reset args
        if (mKeyValueMaxlen != ROUND_UP(mKVCacheCLManager->maxLength(), 4)) {
            mKeyValueMaxlen = ROUND_UP(mKVCacheCLManager->maxLength(), 4);
        }
#ifndef ENABLE_OPENCL_TIME_PROFILER
        if (mOpenCLBackend->isUseRecordQueue()) {
            if (mFlashPrefill) {
                mRgUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->key()))();
                mRgVUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->value()))();
                mFaUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->key()))();
                mFaUpdateInfo.update_kernel_args[1].arg_value = &(*(mKVCacheCLManager->value()))();
            } else {
                // Long prefill folds key and value into one rearrange_qkv launch.
                mRgUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->key()))();
                mRgUpdateInfo.update_kernel_args[1].arg_value = &(*(mKVCacheCLManager->value()))();
            }
        } else {
#endif
            if (mFlashPrefill) {
                cl_int ret = CL_SUCCESS;
                ret |= mKernel_rearrange->get().setArg(4, *mKVCacheCLManager->key());
                ret |= mKernel_rearrange->get().setArg(5, mPastKvSeqlen);
                ret |= mKernel_rearrange->get().setArg(6, mKeyValueMaxlen);
                ret |= mKernel_rearrangeV->get().setArg(4, *mKVCacheCLManager->value());
                ret |= mKernel_rearrangeV->get().setArg(5, mPastKvSeqlen);
                ret |= mKernel_rearrangeV->get().setArg(6, mKeyValueMaxlen);
                ret |= mKernel_fa->get().setArg(1, *mKVCacheCLManager->key());
                ret |= mKernel_fa->get().setArg(2, *mKVCacheCLManager->value());
                ret |= mKernel_fa->get().setArg(7, mKvSeqlen);
                ret |= mKernel_fa->get().setArg(9, mKeyValueMaxlen);
                MNN_CHECK_CL_SUCCESS(ret, "reSetArg flash_attention_prefill");
            } else {
                // rearrange key value
                cl_int ret = CL_SUCCESS;
                ret |= mKernel_rearrange_vec[0]->get().setArg(9, *mKVCacheCLManager->key());
                ret |= mKernel_rearrange_vec[0]->get().setArg(10, *mKVCacheCLManager->value());
                ret |= mKernel_rearrange_vec[0]->get().setArg(14, mKeyValueMaxlen);
                MNN_CHECK_CL_SUCCESS(ret, "reSetArg rearrange_k");
            }
#ifndef ENABLE_OPENCL_TIME_PROFILER
        }
#endif
        return NO_ERROR;
    }

    // Decode
    mKeyValueMaxlen = ROUND_UP(mKVCacheCLManager->maxLength(), 4);
    mFdNumChunk = UP_DIV(mKvSeqlen, mFdChunk);
    if (mKvSeqlen > mDecodeTmpMaxlen) {
        mDecodeTmpMaxlen = mKeyValueMaxlen;
        mFdMaxChunk = UP_DIV(mDecodeTmpMaxlen, mFdChunk);
        mTempPartialO.reset(Tensor::createDevice<int32_t>({numHead * mFdMaxChunk * headDim}));
        mTempPartialML.reset(Tensor::createDevice<int32_t>({numHead * mFdMaxChunk * 2}));
        mOpenCLBackend->onAcquireBuffer(mTempPartialO.get(), Backend::DYNAMIC_IN_EXECUTION);
        mOpenCLBackend->onAcquireBuffer(mTempPartialML.get(), Backend::DYNAMIC_IN_EXECUTION);
        mOpenCLBackend->onReleaseBuffer(mTempPartialO.get(), Backend::DYNAMIC_IN_EXECUTION);
        mOpenCLBackend->onReleaseBuffer(mTempPartialML.get(), Backend::DYNAMIC_IN_EXECUTION);
    }
    mFdPartialGlobal_size[0] = ROUND_UP((uint32_t)(mFdWgSize * mFdNumChunk), (uint32_t)mFdWgSize);
    mGwsFdPartial[0] = mFdPartialGlobal_size[0];
#ifndef ENABLE_OPENCL_TIME_PROFILER
    if (mOpenCLBackend->isUseRecordQueue()) {
        mFdPartialUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->key()))();
        mFdPartialUpdateInfo.update_kernel_args[1].arg_value = &(*(mKVCacheCLManager->value()))();
        mFdPartialUpdateInfo.update_kernel_args[2].arg_value = &openCLDeferBuffer(mTempPartialO.get())();
        mFdPartialUpdateInfo.update_kernel_args[3].arg_value = &openCLDeferBuffer(mTempPartialML.get())();
        mFdPartialUpdateInfo.update_kernel_args[7].arg_value = &(*(mKVCacheCLManager->key()))();
        mFdPartialUpdateInfo.update_kernel_args[8].arg_value = &(*(mKVCacheCLManager->value()))();
        mFdReduceUpdateInfo.update_kernel_args[0].arg_value = &openCLDeferBuffer(mTempPartialO.get())();
        mFdReduceUpdateInfo.update_kernel_args[1].arg_value = &openCLDeferBuffer(mTempPartialML.get())();
    } else {
#endif
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_fdPartial->get().setArg(1, *mKVCacheCLManager->key());
        ret |= mKernel_fdPartial->get().setArg(2, *mKVCacheCLManager->value());
        ret |= mKernel_fdPartial->get().setArg(3, openCLDeferBuffer(mTempPartialO.get()));
        ret |= mKernel_fdPartial->get().setArg(4, openCLDeferBuffer(mTempPartialML.get()));
        ret |= mKernel_fdPartial->get().setArg(6, mKvSeqlen);
        ret |= mKernel_fdPartial->get().setArg(7, mKeyValueMaxlen);
        ret |= mKernel_fdPartial->get().setArg(9, mFdNumChunk);
        ret |= mKernel_fdPartial->get().setArg(13, *mKVCacheCLManager->key());
        ret |= mKernel_fdPartial->get().setArg(14, *mKVCacheCLManager->value());
        ret |= mKernel_fdReduce->get().setArg(0, openCLDeferBuffer(mTempPartialO.get()));
        ret |= mKernel_fdReduce->get().setArg(1, openCLDeferBuffer(mTempPartialML.get()));
        ret |= mKernel_fdReduce->get().setArg(4, mFdNumChunk);
        MNN_CHECK_CL_SUCCESS(ret, "reSetArg flash_decode");
#ifndef ENABLE_OPENCL_TIME_PROFILER
    }
#endif
    return NO_ERROR;
}

int AttentionBufExecution::getLocalSize(int size, int maxGroupSize) {
    int local_size = 1;
    while (local_size * 2 <= maxGroupSize && local_size * 2 <= size) {
        local_size *= 2;
    }
    return local_size;
}

ErrorCode AttentionBufExecution::longPrefillResize(const std::vector<Tensor*>& inputs,
                                                   const std::vector<Tensor*>& outputs) {
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto shape = query->shape();

    int batch = shape[0];
    int seqlen = shape[1];
    int kvInputLen = key->shape()[1];
    int numHead = shape[2];
    int kvNumHead = key->shape()[2];
    int headDim = shape[3];
    int group_size = numHead / kvNumHead;
    float scale = (mAttnScale == 0.0f) ? (1.0f / sqrt(headDim)) : mAttnScale;
    int maskQlen = seqlen;
    int maskKvlen = mKvSeqlen;
    if (mHasMask) {
        auto mask = inputs[3];
        auto maskShape = mask->shape();
        int dim = mask->dimensions();
        MNN_ASSERT(dim >= 2);
        maskQlen = maskShape[dim - 2];
        maskKvlen = maskShape[dim - 1];
    }

    mAlignQ = 32;
    mAlignKV = 32;
    mAlignHDK = 4;
    mAlignHDN = 32;

    float useMemorySize =
        1.0 * ROUND_UP(seqlen, mAlignQ) / 1024.0 * ROUND_UP(mKvSeqlen, mAlignKV) / 1024.0 * batch * numHead;
    // elementSize larger than 32M
    if (useMemorySize > 32.0) {
        mQseqSplitNum = useMemorySize >= 256.0 ? 8 : ((useMemorySize < 128.0) ? 2 : 4);
    }
    // splitPiecesSize need aligned to 32, make sure XgemmBatched globalsize be divisible by localsize
    int splitPiecesSize = ROUND_UP(seqlen, mAlignQ) / mQseqSplitNum;
    while ((splitPiecesSize % 32) != 0) {
        mAlignQ *= 2;
        splitPiecesSize = ROUND_UP(seqlen, mAlignQ) / mQseqSplitNum;
    }

    mKernel_rearrange_vec.resize(1);
    mGwsRearrgVec.resize(1);
    mLwsRearrgVec.resize(1);
    mKernel_mask_vec.resize(1);
    mGwsMaskVec.resize(1);
    mLwsMaskVec.resize(1);
    mKernel_qk_vec.resize(mQseqSplitNum);
    mGwsQkVec.resize(mQseqSplitNum);
    mLwsQkVec.resize(mQseqSplitNum);
    mKernel_softmax_vec.resize(mQseqSplitNum);
    mGwsSoftMaxVec.resize(mQseqSplitNum);
    mLwsSoftMaxVec.resize(mQseqSplitNum);
    mKernel_trans_vec.resize(mQseqSplitNum);
    mGwsTransVec.resize(mQseqSplitNum);
    mLwsTransVec.resize(mQseqSplitNum);
    mKernel_qkv_vec.resize(mQseqSplitNum);
    mGwsQkvVec.resize(mQseqSplitNum);
    mLwsQkvVec.resize(mQseqSplitNum);
    mKernel_clip_vec.resize(1);
    mGwsClipVec.resize(1);
    mLwsClipVec.resize(1);

    mTempQ.reset(
        Tensor::createDevice<float>({ROUND_UP(seqlen, mAlignQ) * ROUND_UP(headDim, mAlignHDK) * batch * numHead}));
    mTempK.reset(Tensor::createDevice<float>(
        {ROUND_UP(mKvSeqlen, mAlignKV) * ROUND_UP(headDim, mAlignHDK) * batch * kvNumHead}));
    mTempV.reset(Tensor::createDevice<float>(
        {ROUND_UP(mKvSeqlen, mAlignKV) * ROUND_UP(headDim, mAlignHDN) * batch * kvNumHead}));
    if (mHasMask) {
        if (mIsAddMask) {
            mTempMask.reset(
                Tensor::createDevice<float>({ROUND_UP(maskQlen, mAlignQ) * ROUND_UP(maskKvlen, mAlignKV) * batch}));
        } else {
            mTempMask.reset(
                Tensor::createDevice<uint32_t>({ROUND_UP(maskQlen, mAlignQ) * ROUND_UP(maskKvlen, mAlignKV) * batch}));
        }
    }
    mTempQK.reset(Tensor::createDevice<float>(
        {ROUND_UP(seqlen, mAlignQ) * ROUND_UP(mKvSeqlen, mAlignKV) * batch * numHead / mQseqSplitNum}));
    mTempSoftMax.reset(Tensor::createDevice<float>(
        {ROUND_UP(seqlen, mAlignQ) * ROUND_UP(mKvSeqlen, mAlignKV) * batch * numHead / mQseqSplitNum}));
    mTempQKV.reset(
        Tensor::createDevice<float>({ROUND_UP(seqlen, mAlignQ) * ROUND_UP(headDim, mAlignHDN) * batch * numHead}));

    mOpenCLBackend->onAcquireBuffer(mTempQ.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempK.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempV.get(), Backend::DYNAMIC);
    if (mHasMask) {
        mOpenCLBackend->onAcquireBuffer(mTempMask.get(), Backend::DYNAMIC);
    }
    mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempQKV.get(), Backend::DYNAMIC);

    mOpenCLBackend->onReleaseBuffer(mTempQ.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mTempK.get(), Backend::DYNAMIC);
    if (mHasMask) {
        mOpenCLBackend->onReleaseBuffer(mTempMask.get(), Backend::DYNAMIC);
    }
    mOpenCLBackend->onReleaseBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mTempV.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mTempQKV.get(), Backend::DYNAMIC);

    // query: [batch, seqLenQ, headNum, headDim] -> mTempQ: [batch*headNum, ROUND_UP(headDim, mAlignHDK),
    // ROUND_UP(seqLenQ, mAlignQ)] key: [batch, seqLenKV/4, headNum/group, headDim, seqLenKV_4] -> mTempK:
    // [batch*headNum/group, ROUND_UP(headDim, mAlignHDK), ROUND_UP(seqLenKV, mAlignKV)] value: [batch, seqLenKV/4,
    // headNum/group, headDim, seqLenKV_4] -> mTempV: [batch*headNum/group, ROUND_UP(seqLenKV, mAlignKV),
    // ROUND_UP(headDim, mAlignHDK] key & value -> pastKey & pastValue (copy)
    int seq_idx = 0;
    // rearrange qkv
    {
        std::set<std::string> buildOption;
        if (TensorUtils::getDescribe(value)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            buildOption.emplace("-DVALUE_C4");
        }
        if ((headDim % 4) != 0) {
            buildOption.emplace("-DHEADDIM_LEAVE");
        }
        // generate cache for every option
        {
            auto option = buildOption;
            auto kernel = runtime->buildKernel("attention_long_layout_buf", "rearrange_qkv", option,
                                               mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        }
        {
            auto option = buildOption;
            option.emplace("-DSEQLEN_LEAVE");
            auto kernel = runtime->buildKernel("attention_long_layout_buf", "rearrange_qkv", option,
                                               mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        }
        if ((seqlen % 4) != 0 || (kvInputLen % 4) != 0) {
            buildOption.emplace("-DSEQLEN_LEAVE");
        }
        if (nullptr != mMeta) {
            buildOption.emplace("-DSAVE_KV");
        }
        int seq_len_pack_q = ROUND_UP(seqlen, mAlignQ);
        int seq_len_pack_kv = ROUND_UP(mKvSeqlen, mAlignKV);

        int head_dim_pack_qk = ROUND_UP(headDim, mAlignHDK);
        int head_dim_pack_v = ROUND_UP(headDim, mAlignHDN);

        int tile[4] = {mAlignQ, mAlignKV, mAlignHDK, mAlignHDN};
        int shape[4] = {seqlen, kvInputLen, numHead, headDim};
        int param[4] = {group_size, batch, 0, 0};
        mKernel_rearrange_vec[seq_idx] = runtime->buildKernel("attention_long_layout_buf", "rearrange_qkv", buildOption,
                                                              mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_rearrange_vec[seq_idx]));

        mGwsRearrgVec[seq_idx] = {
            static_cast<uint32_t>(ALIMAX(UP_DIV(seq_len_pack_q, 4), UP_DIV(seq_len_pack_kv, 4))),
            static_cast<uint32_t>(ALIMAX(UP_DIV(head_dim_pack_qk, 4), UP_DIV(head_dim_pack_v, 4))),
            static_cast<uint32_t>(batch * numHead)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, mGwsRearrgVec[seq_idx][0]);
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, mGwsRearrgVec[seq_idx][1]);
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, mGwsRearrgVec[seq_idx][2]);
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, openCLBuffer(query));
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, openCLBuffer(key));
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, openCLBuffer(value));
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempQ.get()));
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempK.get()));
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempV.get()));
        if (nullptr != mMeta) {
            ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, *mKVCacheCLManager->key());
            ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, *mKVCacheCLManager->value());
        }
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, tile);
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, shape);
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, param);
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, mKeyValueMaxlen);

        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_qkv");
        mLwsRearrgVec[seq_idx] = localWS3DDefault(mGwsRearrgVec[seq_idx], maxWorkGroupSize, runtime, "rearrange_qkv",
                                                  mKernel_rearrange_vec[seq_idx], mOpenCLBackend->getCLTuneLevel(),
                                                  "attention_long_layout_buf")
                                     .first;
        mGwsRearrgVec[seq_idx][0] =
            ROUND_UP(mGwsRearrgVec[seq_idx][0], std::max((uint32_t)1, mLwsRearrgVec[seq_idx][0]));
        mGwsRearrgVec[seq_idx][1] =
            ROUND_UP(mGwsRearrgVec[seq_idx][1], std::max((uint32_t)1, mLwsRearrgVec[seq_idx][1]));
        mGwsRearrgVec[seq_idx][2] =
            ROUND_UP(mGwsRearrgVec[seq_idx][2], std::max((uint32_t)1, mLwsRearrgVec[seq_idx][2]));
        if (nullptr != mMeta) {
            mRgUpdateInfo.update_kernel_args.push_back({0, 9, sizeof(cl_mem), &(*(mKVCacheCLManager->key()))()});
            mRgUpdateInfo.update_kernel_args.push_back({0, 10, sizeof(cl_mem), &(*(mKVCacheCLManager->value()))()});
        }
        mRgUpdateInfo.update_kernel_args.push_back({0, 14, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
        mOpRecordUpdateInfo.emplace_back(&mRgUpdateInfo);
        mOpenCLBackend->recordKernel3d(mKernel_rearrange_vec[seq_idx], mGwsRearrgVec[seq_idx], mLwsRearrgVec[seq_idx],
                                       &mRgUpdateInfo);
    }

    // mask rearaange
    if (mHasMask) {
        std::set<std::string> buildOption;

        int seq_len_pack_q = ROUND_UP(maskQlen, mAlignQ);
        int seq_len_pack_kv = ROUND_UP(maskKvlen, mAlignKV);
        int shape[4] = {seqlen, maskKvlen, mAlignQ, mAlignKV};

        mKernel_mask_vec[seq_idx] = runtime->buildKernel("attention_long_layout_buf", "rearrange_mask", buildOption,
                                                         mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_mask_vec[seq_idx]));

        mGwsMaskVec[seq_idx] = {static_cast<uint32_t>(UP_DIV(seq_len_pack_q, 4)),
                                static_cast<uint32_t>(UP_DIV(seq_len_pack_kv, 4)), static_cast<uint32_t>(batch)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_mask_vec[seq_idx]->get().setArg(index++, mGwsMaskVec[seq_idx][0]);
        ret |= mKernel_mask_vec[seq_idx]->get().setArg(index++, mGwsMaskVec[seq_idx][1]);
        ret |= mKernel_mask_vec[seq_idx]->get().setArg(index++, mGwsMaskVec[seq_idx][2]);
        ret |= mKernel_mask_vec[seq_idx]->get().setArg(index++, openCLBuffer(inputs[3]));
        ret |= mKernel_mask_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempMask.get()));
        ret |= mKernel_mask_vec[seq_idx]->get().setArg(index++, shape);

        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_mask");
        mLwsMaskVec[seq_idx] =
            localWS3DDefault(mGwsMaskVec[seq_idx], maxWorkGroupSize, runtime, "rearrange_mask",
                             mKernel_mask_vec[seq_idx], mOpenCLBackend->getCLTuneLevel(), "attention_long_layout_buf")
                .first;
        mGwsMaskVec[seq_idx][0] = ROUND_UP(mGwsMaskVec[seq_idx][0], std::max((uint32_t)1, mLwsMaskVec[seq_idx][0]));
        mGwsMaskVec[seq_idx][1] = ROUND_UP(mGwsMaskVec[seq_idx][1], std::max((uint32_t)1, mLwsMaskVec[seq_idx][1]));
        mGwsMaskVec[seq_idx][2] = ROUND_UP(mGwsMaskVec[seq_idx][2], std::max((uint32_t)1, mLwsMaskVec[seq_idx][2]));
        mOpenCLBackend->recordKernel3d(mKernel_mask_vec[seq_idx], mGwsMaskVec[seq_idx], mLwsMaskVec[seq_idx]);
    }

    for (int seq_idx = 0; seq_idx < mQseqSplitNum; seq_idx++) {
        // qk matmul
        {
            // Q : [batch*headNum, ROUND_UP(headDim, mAlignHDK), ROUND_UP(seqLenQ, mAlignQ) / mQseqSplitNum] -> [B, K,
            // M] K : [batch*headNum/group, ROUND_UP(headDim, mAlignHDK), ROUND_UP(seqLenKV, mAlignKV)] -> [B, K, N] QV:
            // [Batch * numHead, ROUND_UP(seqLenQ, mAlignQ) / mQseqSplitNum, ROUND_UP(seqLenKV, mAlignKV)]   -> [B, M,
            // N]
            int loop = batch * numHead;
            int e_pack = ROUND_UP(seqlen, mAlignQ);
            int e_pack_piece = e_pack / mQseqSplitNum;
            int h_pack = ROUND_UP(mKvSeqlen, mAlignKV);
            int l_pack = ROUND_UP(headDim, mAlignHDK);

            std::set<std::string> buildOptions;

            int biasType = 0;
            if (mIsAddMask) {
                biasType = 2;
            } else if (mHasMask) {
                biasType = 5; // int value mask
            }
            uint32_t layout = 14; // 10 means mix-precision, 4 means layout
            auto param = getGemmParams({(uint32_t)e_pack_piece, (uint32_t)h_pack, (uint32_t)l_pack, layout,
                                        (uint32_t)loop, (uint32_t)(biasType + 10 * (group_size - 1))},
                                       mOpenCLBackend->getOpenCLRuntime(), mOpenCLBackend->getPrecision(),
                                       mOpenCLBackend->getCLTuneLevel());

            int KWG = param[0], KWI = param[1], MDIMA = param[2], MDIMC = param[3], MWG = param[4], NDIMB = param[5],
                NDIMC = param[6], NWG = param[7], SA = param[8], SB = param[9], STRM = param[10], STRN = param[11],
                VWM = param[12], VWN = param[13];
            buildOptions.emplace("-DKWG=" + std::to_string(KWG));
            buildOptions.emplace("-DKWI=" + std::to_string(KWI));
            buildOptions.emplace("-DMDIMA=" + std::to_string(MDIMA));
            buildOptions.emplace("-DMDIMC=" + std::to_string(MDIMC));
            buildOptions.emplace("-DMWG=" + std::to_string(MWG));
            buildOptions.emplace("-DNDIMB=" + std::to_string(NDIMB));
            buildOptions.emplace("-DNDIMC=" + std::to_string(NDIMC));
            buildOptions.emplace("-DNWG=" + std::to_string(NWG));
            buildOptions.emplace("-DSA=" + std::to_string(SA));
            buildOptions.emplace("-DSB=" + std::to_string(SB));
            buildOptions.emplace("-DSTRM=" + std::to_string(STRM));
            buildOptions.emplace("-DSTRN=" + std::to_string(STRN));
            buildOptions.emplace("-DVWM=" + std::to_string(VWM));
            buildOptions.emplace("-DVWN=" + std::to_string(VWN));
            if (layout >= 4) {
                buildOptions.emplace("-DOUTPUTMN");
            }

            int tileM = MWG;
            int tileN = NWG;
            int localM = MDIMC;
            int localN = NDIMC;

            if (mOpenCLBackend->getOpenCLRuntime()->getGpuType() == GpuType::ADRENO) {
                buildOptions.emplace("-DUSE_CL_MAD=1");
                buildOptions.emplace("-DRELAX_WORKGROUP_SIZE=1");
            }
            buildOptions.emplace("-DONLY_HAVE_ALPHA");
            if (biasType >= 1) {
                buildOptions.emplace("-DBIAS_TYPE=" + std::to_string(biasType));
            }

            buildOptions.emplace("-DPRECISION_COMPUTE=float -DCONVERT_PRECISION_COMPUTE=convert_float");
            buildOptions.emplace("-DPRECISION_COMPUTE2=float2 -DCONVERT_PRECISION_COMPUTE2=convert_float2");
            buildOptions.emplace("-DPRECISION_COMPUTE4=float4 -DCONVERT_PRECISION_COMPUTE4=convert_float4");
            buildOptions.emplace("-DPRECISION_COMPUTE8=float8 -DCONVERT_PRECISION_COMPUTE8=convert_float8");
            buildOptions.emplace("-DPRECISION_COMPUTE16=float16 -DCONVERT_PRECISION_COMPUTE16=convert_float16");

            mKernel_qk_vec[seq_idx] = mOpenCLBackend->getOpenCLRuntime()->buildKernel(
                "matmul_params_buf", "XgemmBatched", buildOptions, mOpenCLBackend->getPrecision());

            int out_per_thread_m = tileM / localM;
            int out_per_thread_n = tileN / localN;

            mGwsQkVec[seq_idx] = {static_cast<uint32_t>(e_pack_piece / out_per_thread_m),
                                  static_cast<uint32_t>(h_pack / out_per_thread_n), static_cast<uint32_t>(loop)};
            mLwsQkVec[seq_idx] = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN), 1};

            float alpha = scale;
            float beta = 0.0f;
            int batch_offset_a = e_pack * l_pack;
            int batch_offset_b = h_pack * l_pack;
            int batch_offset_c = e_pack_piece * h_pack;

            int batch_offset[4] = {batch_offset_a, batch_offset_b, batch_offset_c, 0};
            int base_ptr_offset[4] = {e_pack_piece * seq_idx, 0, 0, batch_offset_c * seq_idx};
            int stride[4] = {e_pack, h_pack, h_pack, h_pack};
            int group[4] = {1, group_size, 1, loop};

            int idx = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, static_cast<int>(e_pack_piece));
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, static_cast<int>(h_pack));
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, static_cast<int>(l_pack));
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, alpha);
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, beta);
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, openCLBuffer(mTempQ.get()));
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, openCLBuffer(mTempK.get()));
            if (mHasMask) {
                // Only when biasType >= 1 does the kernel have the egm (bias) parameter;
                // maskless builds have no BIAS_TYPE, and setting an extra arg fails with
                // CL_INVALID_ARG_INDEX, silently skipping the QK matmul.
                ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, openCLBuffer(mTempMask.get()));
            }
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, openCLBuffer(mTempQK.get()));
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, batch_offset);
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, base_ptr_offset);
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, stride);
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, group);
            MNN_CHECK_CL_SUCCESS(ret, "setArg Self-Attention batchmatmul qk Kernel");
            mOpenCLBackend->recordKernel3d(mKernel_qk_vec[seq_idx], mGwsQkVec[seq_idx], mLwsQkVec[seq_idx]);
        }

        // softmax
        {
            // QV:     [Batch * numHead, ROUND_UP(seqLenQ, mAlignQ) / mQseqSplitNum, ROUND_UP(seqLenKV, mAlignKV)]
            // Sotmax: [Batch * numHead, ROUND_UP(seqLenQ, mAlignQ) / mQseqSplitNum, ROUND_UP(seqLenKV, mAlignKV)]
            // axis  : 2 (last dim)
            int softmaxShape[4];
            softmaxShape[0] = batch * numHead;
            softmaxShape[1] = ROUND_UP(seqlen, mAlignQ) / mQseqSplitNum;
            softmaxShape[2] = ROUND_UP(mKvSeqlen, mAlignKV);

            auto MaxLocalSize =
                std::min(std::min(runtime->getMaxWorkItemSizes()[0], mMaxWorkGroupSize), static_cast<uint32_t>(256));
            int localSize = 64;

            std::set<std::string> buildOption;
            buildOption.emplace("-DSOFTMAX_LOCAL_SIZE=" + std::to_string(localSize));

            mKernel_softmax_vec[seq_idx] = runtime->buildKernel("self_attention_buf", "softmax_inside", buildOption,
                                                                mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
            mGwsSoftMaxVec[seq_idx] = {static_cast<uint32_t>(localSize), static_cast<uint32_t>(softmaxShape[1]),
                                       static_cast<uint32_t>(softmaxShape[0])};

            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_softmax_vec[seq_idx]->get().setArg(index++, mGwsSoftMaxVec[seq_idx][0]);
            ret |= mKernel_softmax_vec[seq_idx]->get().setArg(index++, mGwsSoftMaxVec[seq_idx][1]);
            ret |= mKernel_softmax_vec[seq_idx]->get().setArg(index++, mGwsSoftMaxVec[seq_idx][2]);
            ret |= mKernel_softmax_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempQK.get()));
            ret |= mKernel_softmax_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempSoftMax.get()));
            ret |= mKernel_softmax_vec[seq_idx]->get().setArg(index++, mKvSeqlen);
            ret |= mKernel_softmax_vec[seq_idx]->get().setArg(index++, softmaxShape);
            MNN_CHECK_CL_SUCCESS(ret, "setArg Attention softmax");

            mLwsSoftMaxVec[seq_idx] = {static_cast<uint32_t>(localSize), 1, 1};
            mOpenCLBackend->recordKernel3d(mKernel_softmax_vec[seq_idx], mGwsSoftMaxVec[seq_idx],
                                           mLwsSoftMaxVec[seq_idx]);
        }
        {
            // Sotmax: [Batch * numHead, ROUND_UP(seqLenQ, mAlignQ) / mQseqSplitNum, ROUND_UP(seqLenKV, mAlignKV)]
            // Trans:  [Batch * numHead, ROUND_UP(seqLenKV, mAlignKV), ROUND_UP(seqLenQ, mAlignQ) / mQseqSplitNum]
            int loop = batch * numHead;
            int transDimW = ROUND_UP(seqlen, mAlignQ) / mQseqSplitNum;
            int transDimH = ROUND_UP(mKvSeqlen, mAlignKV);

            std::set<std::string> buildOptions;
            mKernel_trans_vec[seq_idx] = runtime->buildKernel("self_attention_buf", "trans_3d_buf", buildOptions,
                                                              mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(
                mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel_trans_vec[seq_idx]));

            mGwsTransVec[seq_idx] = {(uint32_t)transDimW / 8, (uint32_t)transDimH / 8, (uint32_t)(loop)};

            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_trans_vec[seq_idx]->get().setArg(index++, mGwsTransVec[seq_idx][0]);
            ret |= mKernel_trans_vec[seq_idx]->get().setArg(index++, mGwsTransVec[seq_idx][1]);
            ret |= mKernel_trans_vec[seq_idx]->get().setArg(index++, mGwsTransVec[seq_idx][2]);
            ret |= mKernel_trans_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempSoftMax.get()));
            ret |= mKernel_trans_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempQK.get()));
            ret |= mKernel_trans_vec[seq_idx]->get().setArg(index++, loop);
            ret |= mKernel_trans_vec[seq_idx]->get().setArg(index++, transDimW);
            ret |= mKernel_trans_vec[seq_idx]->get().setArg(index++, transDimH);
            MNN_CHECK_CL_SUCCESS(ret, "setArg Attention transpose");
            mLwsTransVec[seq_idx] =
                localWS3DDefault(mGwsTransVec[seq_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(),
                                 "trans_3d_buf", mKernel_trans_vec[seq_idx], mOpenCLBackend->getCLTuneLevel(),
                                 "self_attention_buf")
                    .first;

            mGwsTransVec[seq_idx][0] =
                ROUND_UP(mGwsTransVec[seq_idx][0], std::max((uint32_t)1, mLwsTransVec[seq_idx][0]));
            mGwsTransVec[seq_idx][1] =
                ROUND_UP(mGwsTransVec[seq_idx][1], std::max((uint32_t)1, mLwsTransVec[seq_idx][1]));
            mGwsTransVec[seq_idx][2] =
                ROUND_UP(mGwsTransVec[seq_idx][2], std::max((uint32_t)1, mLwsTransVec[seq_idx][2]));

            mOpenCLBackend->recordKernel3d(mKernel_trans_vec[seq_idx], mGwsTransVec[seq_idx], mLwsTransVec[seq_idx]);
        }

        // qk * value
        {
            // Trans: [Batch * numHead, ROUND_UP(seqLenKV, mAlignKV), ROUND_UP(seqLenQ, mAlignQ) / mQseqSplitNum]   ->
            // [B, K, M] V :     [Batch * numHead / group, ROUND_UP(seqLenKV, mAlignKV), ROUND_UP(headDim, mAlignHDN)]
            // -> [B, K, N] QKV :   [Batch * numHead, ROUND_UP(headDim, mAlignHDN), ROUND_UP(seqLenQ, mAlignQ) /
            // mQseqSplitNum] -> [B, N, M]

            int loop = batch * numHead;
            int e_pack = ROUND_UP(seqlen, mAlignQ);
            int e_pack_piece = e_pack / mQseqSplitNum;
            int l_pack = ROUND_UP(mKvSeqlen, mAlignKV);
            int h_pack = ROUND_UP(headDim, mAlignHDN);

            std::set<std::string> buildOptions;

            uint32_t layout = 0;
            // NOTE: mTempV holds only batch*kvNumHead heads (GQA). The tuning kernel must divide the
            // batch index by group_size when indexing V, otherwise it reads out of bounds (Mali
            // GROUP_ERROR_FATAL). Encode group_size into gemmSize[5] as the qk path does (biasType == 0 here).
            auto param = getGemmParams({(uint32_t)e_pack_piece, (uint32_t)h_pack, (uint32_t)l_pack, layout,
                                        (uint32_t)loop, (uint32_t)(10 * (group_size - 1))},
                                       mOpenCLBackend->getOpenCLRuntime(), mOpenCLBackend->getPrecision(),
                                       mOpenCLBackend->getCLTuneLevel());

            int KWG = param[0], KWI = param[1], MDIMA = param[2], MDIMC = param[3], MWG = param[4], NDIMB = param[5],
                NDIMC = param[6], NWG = param[7], SA = param[8], SB = param[9], STRM = param[10], STRN = param[11],
                VWM = param[12], VWN = param[13];
            buildOptions.emplace("-DKWG=" + std::to_string(KWG));
            buildOptions.emplace("-DKWI=" + std::to_string(KWI));
            buildOptions.emplace("-DMDIMA=" + std::to_string(MDIMA));
            buildOptions.emplace("-DMDIMC=" + std::to_string(MDIMC));
            buildOptions.emplace("-DMWG=" + std::to_string(MWG));
            buildOptions.emplace("-DNDIMB=" + std::to_string(NDIMB));
            buildOptions.emplace("-DNDIMC=" + std::to_string(NDIMC));
            buildOptions.emplace("-DNWG=" + std::to_string(NWG));
            buildOptions.emplace("-DSA=" + std::to_string(SA));
            buildOptions.emplace("-DSB=" + std::to_string(SB));
            buildOptions.emplace("-DSTRM=" + std::to_string(STRM));
            buildOptions.emplace("-DSTRN=" + std::to_string(STRN));
            buildOptions.emplace("-DVWM=" + std::to_string(VWM));
            buildOptions.emplace("-DVWN=" + std::to_string(VWN));
            if (layout >= 4) {
                buildOptions.emplace("-DOUTPUTMN");
            }

            int tileM = MWG;
            int tileN = NWG;
            int localM = MDIMC;
            int localN = NDIMC;

            if (mOpenCLBackend->getOpenCLRuntime()->getGpuType() == GpuType::ADRENO) {
                buildOptions.emplace("-DUSE_CL_MAD=1");
                buildOptions.emplace("-DRELAX_WORKGROUP_SIZE=1");
            }

            mKernel_qkv_vec[seq_idx] = mOpenCLBackend->getOpenCLRuntime()->buildKernel(
                "matmul_params_buf", "XgemmBatched", buildOptions, mOpenCLBackend->getPrecision());

            int out_per_thread_m = tileM / localM;
            int out_per_thread_n = tileN / localN;

            mGwsQkvVec[seq_idx] = {static_cast<uint32_t>(e_pack_piece / out_per_thread_m),
                                   static_cast<uint32_t>(h_pack / out_per_thread_n), static_cast<uint32_t>(loop)};
            mLwsQkvVec[seq_idx] = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN), 1};

            float alpha = 1.0f;
            float beta = 0.0f;
            int batch_offset_a = e_pack_piece * l_pack;
            int batch_offset_b = h_pack * l_pack;
            int batch_offset_c = e_pack * h_pack;
            int batch_offset[4] = {batch_offset_a, batch_offset_b, batch_offset_c, 0};
            int base_ptr_offset[4] = {0, 0, e_pack_piece * seq_idx, 0};
            int stride[4] = {e_pack_piece, h_pack, e_pack, h_pack};
            int group[4] = {1, group_size, 1, loop};

            int idx = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, static_cast<int>(e_pack_piece));
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, static_cast<int>(h_pack));
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, static_cast<int>(l_pack));
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, alpha);
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, beta);
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, openCLBuffer(mTempQK.get()));
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, openCLBuffer(mTempV.get()));
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, openCLBuffer(mTempQKV.get()));
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, batch_offset);
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, base_ptr_offset);
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, stride);
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, group);
            MNN_CHECK_CL_SUCCESS(ret, "setArg Self-Attention batchmatmul qkv Kernel");
            mOpenCLBackend->recordKernel3d(mKernel_qkv_vec[seq_idx], mGwsQkvVec[seq_idx], mLwsQkvVec[seq_idx]);
        }
    }

    seq_idx = 0;
    // transpose to output
    {
        // QKV :   [Batch * numHead, ROUND_UP(headDim, mAlignHDN), ROUND_UP(seqLenQ, mAlignQ)] -> [B, N, M]
        // output: [batch, seqLenQ/4, headNum, headDim, seqLenQ_4]
        std::set<std::string> buildOption;
        if (mOutputC4) {
            buildOption.emplace("-DATTENTION_C4");
        }

        mKernel_clip_vec[seq_idx] =
            runtime->buildKernel("attention_long_layout_buf", "qkv_transpose_output", buildOption,
                                 mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_clip_vec[seq_idx]));

        mGwsClipVec[seq_idx] = {static_cast<uint32_t>(UP_DIV(seqlen, 4)), static_cast<uint32_t>(UP_DIV(headDim, 4)),
                                static_cast<uint32_t>(batch * numHead)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, mGwsClipVec[seq_idx][0]);
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, mGwsClipVec[seq_idx][1]);
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, mGwsClipVec[seq_idx][2]);
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempQKV.get()));
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, openCLBuffer(outputs[0]));
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, mAlignQ);
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, mAlignHDN);
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, seqlen);
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, numHead);
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, headDim);
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, batch);

        mLwsClipVec[seq_idx] =
            localWS3DDefault(mGwsClipVec[seq_idx], maxWorkGroupSize, runtime, "qkv_transpose_output",
                             mKernel_clip_vec[seq_idx], mOpenCLBackend->getCLTuneLevel(), "attention_long_layout_buf")
                .first;
        mGwsClipVec[seq_idx][0] = ROUND_UP(mGwsClipVec[seq_idx][0], std::max((uint32_t)1, mLwsClipVec[seq_idx][0]));
        mGwsClipVec[seq_idx][1] = ROUND_UP(mGwsClipVec[seq_idx][1], std::max((uint32_t)1, mLwsClipVec[seq_idx][1]));
        mGwsClipVec[seq_idx][2] = ROUND_UP(mGwsClipVec[seq_idx][2], std::max((uint32_t)1, mLwsClipVec[seq_idx][2]));

        MNN_CHECK_CL_SUCCESS(ret, "setArg qkv_transpose_output");
        mOpenCLBackend->recordKernel3d(mKernel_clip_vec[seq_idx], mGwsClipVec[seq_idx], mLwsClipVec[seq_idx]);
    }
    mOpenCLBackend->endRecord(mRecording);

    return NO_ERROR;
}

// Select compile-time tiling that satisfies every kernel phase: the workgroup covers all
// output units, is a multiple of tileQ, and maps exactly to the QK micro-tiles. Keep smaller
// legal groups as fallbacks because register pressure may lower the kernel-specific limit.
bool AttentionBufExecution::flashPrefillEligible(const std::vector<Tensor*>& inputs,
                                                 const std::vector<Tensor*>& outputs) {
    auto shape = inputs[0]->shape();
    int seqlen = shape[1];
    int headDim = shape[3];
    if (headDim < 1 || seqlen < 1) {
        return false;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const int maxWgSize = std::min((int)mMaxWorkGroupSize, 256);
    const uint64_t maxLocal = runtime->getMaxLocalMem();
    // lq is FLOAT and ls is COMPUTE_FLOAT, and the two widths only agree outside the Normal
    // level, which stores in half but computes in float. See the three buildOptionsStr
    // branches in OpenCLRuntime::buildKernelWithCache.
    const auto precision = mOpenCLBackend->getPrecision();
    const int floatBytes =
        (BackendConfig::Precision_Normal == precision || BackendConfig::Precision_Low == precision) ? 2 : 4;
    const int computeBytes = (BackendConfig::Precision_Low == precision) ? 2 : 4;

    std::vector<FaTiling> legal;
    int bestIdx = -1, bestScore = 1 << 30;
    // Descending tileQ, so that a tie keeps the larger tile: it reuses each kv entry over more
    // query rows, and a group no larger than the phase-3 unit count also compiles out the
    // surplus-lane guard.
    for (int tileQ : {64, 32, 16, 8, 4}) {
        const int p3Units = tileQ / 4 * UP_DIV(headDim, 8);
        // Smallest legal group: a multiple of tileQ that covers every phase-3 unit. Then the
        // wave-aligned one. tileQ is a power of two, so lcm(tileQ, 32) is just the larger of
        // the two and both stay multiples of tileQ.
        const int minWg = ROUND_UP(p3Units, tileQ);
        const int waveAligned = ROUND_UP(p3Units, std::max(tileQ, 32));
        for (int wg : {waveAligned, minWg}) {
            // wg >= tileQ is what keeps the softmax phase's FA_TPR at least 1.
            if (wg > maxWgSize || wg < tileQ) {
                continue;
            }
            // Makes phase 1's (tileQ / 4) x (tileKV / 4) micro-tiles map exactly onto the
            // workgroup, and gives every softmax work-item 16 kv entries.
            const int tileKV = 16 * wg / tileQ;
            // lq[ROUND_UP(headDim,4) * tileQ] FLOAT, ls[tileKV * tileQ] COMPUTE_FLOAT, then
            // lred[tileQ * tpr] + ll[tileQ] + la[tileQ] as float. ROUND_UP because phase 0 fills
            // lq four dim rows at a time, so the dim axis is padded when headDim % 4 != 0.
            const uint64_t localBytes = (uint64_t)ROUND_UP(headDim, 4) * tileQ * floatBytes +
                                        (uint64_t)tileKV * tileQ * computeBytes +
                                        (uint64_t)(wg / tileQ + 3) * tileQ * 4;
            if (localBytes > maxLocal) {
                continue;
            }
            // Prefer a workgroup near 64 lanes, and a multiple of the wave size. The penalty is
            // what keeps a sub-wave group -- only ever reached through minWg -- out of the
            // preferred slot while still leaving it available as a fallback.
            const int score = (wg > 64 ? wg - 64 : 64 - wg) + ((wg % 32) ? 1000 : 0);
            if (score < bestScore) {
                bestScore = score;
                bestIdx = (int)legal.size();
            }
            legal.push_back({tileQ, wg, tileKV});
            if (wg == minWg) {
                break; // waveAligned == minWg, do not enter it twice
            }
        }
    }
    mFaTilings.clear();
    if (bestIdx < 0) {
        return false;
    }
    mFaTilings.push_back(legal[bestIdx]);
    // Fallbacks, for the kernel-level work-group check in flashPrefillResize. Only smaller
    // groups: the local memory a tiling needs scales with it but the register pressure does not,
    // so a kernel that refuses this group size will refuse every larger one too. Descending, to
    // give up as little of the group as the driver forces.
    for (int wgLimit = mFaTilings[0].wgSize; wgLimit > 0;) {
        int pick = -1;
        for (int i = 0; i < (int)legal.size(); ++i) {
            if (legal[i].wgSize < wgLimit && (pick < 0 || legal[i].wgSize > legal[pick].wgSize)) {
                pick = i;
            }
        }
        if (pick < 0) {
            break;
        }
        wgLimit = legal[pick].wgSize;
        mFaTilings.push_back(legal[pick]);
    }
    mFaTileQ = mFaTilings[0].tileQ;
    mFaWgSize = mFaTilings[0].wgSize;
    mFaTileKV = mFaTilings[0].tileKV;
    return true;
}

std::set<std::string> AttentionBufExecution::flashPrefillBuildOptions(int headDim, int groupSize) const {
    std::set<std::string> buildOption;
    if (mIsAddMask) {
        buildOption.emplace("-DADD_MASK");
    } else if (mHasMask) {
        buildOption.emplace("-DSET_MASK");
    } else {
        buildOption.emplace("-DDEFAULT_MASK");
    }
    buildOption.emplace("-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize));
    buildOption.emplace("-DFA_TILE_Q=" + std::to_string(mFaTileQ));
    buildOption.emplace("-DFA_TILE_KV=" + std::to_string(mFaTileKV));
    buildOption.emplace("-DFA_HEAD_DIM=" + std::to_string(headDim));
    buildOption.emplace("-DFA_WG_SIZE=" + std::to_string(mFaWgSize));
    if (mOutputC4) {
        buildOption.emplace("-DATTENTION_C4");
    }
    return buildOption;
}

ErrorCode AttentionBufExecution::flashPrefillResize(const std::vector<Tensor*>& inputs,
                                                    const std::vector<Tensor*>& outputs) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto shape = query->shape();

    int batch = shape[0];
    int seqlen = shape[1];
    int kvInputLen = key->shape()[1];
    int numHead = shape[2];
    int kvNumHead = key->shape()[2];
    int headDim = shape[3];
    int groupSize = numHead / kvNumHead;
    float scale = (mAttnScale == 0.0f) ? (1.0f / sqrt(headDim)) : mAttnScale;
    int maskKvlen = mKvSeqlen;
    int maskQlen = seqlen;
    int maskBatchStride = 0;
    if (mHasMask) {
        auto mask = inputs[3];
        int dim = mask->dimensions();
        maskKvlen = mask->shape()[dim - 1];
        maskQlen = mask->shape()[dim - 2];
        // The plane is [batch, 1, maskQlen, maskKvlen] when it is per batch and broadcast over
        // batch otherwise; elementSize tells the two apart without having to guess which leading
        // axis carries the batch. Zero stride makes every batch read the one plane.
        if (mask->elementSize() == (int)batch * maskQlen * maskKvlen) {
            maskBatchStride = maskQlen * maskKvlen;
        }
    }

    // flashPrefillEligible sized the work groups against the device limit, but a kernel's own
    // CL_KERNEL_WORK_GROUP_SIZE can be lower once its register use is known, and the local size
    // here is not clamped -- that would be CL_INVALID_WORK_GROUP_SIZE at enqueue. Probe now,
    // before anything is recorded or allocated: backing out later would leave the rearrange
    // kernels sitting in the recording, to be replayed on top of the fallback path.
    //
    // A driver that refuses the preferred group still gets flash, on the next tiling down, rather
    // than dropping to short prefill. The first entry is the one the score picked, so the common
    // case builds exactly the program it always did and the retries never compile.
    {
        bool accepted = false;
        for (const auto& tiling : mFaTilings) {
            mFaTileQ = tiling.tileQ;
            mFaWgSize = tiling.wgSize;
            mFaTileKV = tiling.tileKV;
            auto probe = runtime->buildKernel("attention_flash_prefill_buf", "flash_attention_prefill",
                                              flashPrefillBuildOptions(headDim, groupSize),
                                              mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
            if (nullptr != probe && (int)runtime->getMaxWorkGroupSize(probe) >= mFaWgSize) {
                accepted = true;
                break;
            }
        }
        if (!accepted) {
            return NOT_SUPPORT;
        }
    }

    cl::Buffer keyBuffer, valueBuffer;
    if (nullptr != mMeta) {
        keyBuffer = *mKVCacheCLManager->key();
        valueBuffer = *mKVCacheCLManager->value();
    } else {
        mTempK.reset(Tensor::createDevice<float>({ROUND_UP(kvInputLen, 4) * ROUND_UP(headDim, 4) * kvNumHead * batch}));
        mTempV.reset(Tensor::createDevice<float>({ROUND_UP(kvInputLen, 4) * ROUND_UP(headDim, 4) * kvNumHead * batch}));
        mOpenCLBackend->onAcquireBuffer(mTempK.get(), Backend::DYNAMIC);
        mOpenCLBackend->onAcquireBuffer(mTempV.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mTempV.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mTempK.get(), Backend::DYNAMIC);
        keyBuffer = openCLBuffer(mTempK.get());
        valueBuffer = openCLBuffer(mTempV.get());
    }

    {
        // rearrange key -> kv cache
        std::set<std::string> buildOption;
        mKernel_rearrange = runtime->buildKernel("attention_kv_rearrange_buf", "rearrange_k", buildOption,
                                                 mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_rearrange));
        mGlobalWorkSizeRearrg = {static_cast<uint32_t>(UP_DIV(kvInputLen, 4)),
                                 static_cast<uint32_t>(UP_DIV(headDim, 4)), static_cast<uint32_t>(kvNumHead * batch)};
        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[0]);
        ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[1]);
        ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[2]);
        ret |= mKernel_rearrange->get().setArg(index++, openCLBuffer(key));
        ret |= mKernel_rearrange->get().setArg(index++, keyBuffer);
        ret |= mKernel_rearrange->get().setArg(index++, mPastKvSeqlen);
        ret |= mKernel_rearrange->get().setArg(index++, mKeyValueMaxlen);
        ret |= mKernel_rearrange->get().setArg(index++, kvInputLen);
        ret |= mKernel_rearrange->get().setArg(index++, kvNumHead);
        ret |= mKernel_rearrange->get().setArg(index++, numHead);
        ret |= mKernel_rearrange->get().setArg(index++, headDim);
        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_k");
        mLocalWorkSizeRearrg =
            localWS3DDefault(mGlobalWorkSizeRearrg, maxWorkGroupSize, runtime, "rearrange_k", mKernel_rearrange,
                             mOpenCLBackend->getCLTuneLevel(), "attention_kv_rearrange_buf",
                             rearrangeKLwsShortlist(runtime->getGpuType()))
                .first;
        mGlobalWorkSizeRearrg[0] = ROUND_UP(mGlobalWorkSizeRearrg[0], std::max((uint32_t)1, mLocalWorkSizeRearrg[0]));
        mGlobalWorkSizeRearrg[1] = ROUND_UP(mGlobalWorkSizeRearrg[1], std::max((uint32_t)1, mLocalWorkSizeRearrg[1]));
        mGlobalWorkSizeRearrg[2] = ROUND_UP(mGlobalWorkSizeRearrg[2], std::max((uint32_t)1, mLocalWorkSizeRearrg[2]));
        if (nullptr != mMeta) {
            mRgUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &(*(mKVCacheCLManager->key()))()});
        }
        mRgUpdateInfo.update_kernel_args.push_back({0, 5, sizeof(mPastKvSeqlen), &mPastKvSeqlen});
        mRgUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
        mOpRecordUpdateInfo.emplace_back(&mRgUpdateInfo);
        mOpenCLBackend->recordKernel3d(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg, &mRgUpdateInfo);
    }
    {
        // rearrange value -> kv cache
        std::set<std::string> buildOption;
        if (TensorUtils::getDescribe(value)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            buildOption.emplace("-DVALUE_C4");
        }
        mKernel_rearrangeV = runtime->buildKernel("attention_kv_rearrange_buf", "rearrange_v", buildOption,
                                                  mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_rearrangeV));
        mGlobalWorkSizeRearrgV = {static_cast<uint32_t>(UP_DIV(headDim, 4)),
                                  static_cast<uint32_t>(UP_DIV(kvInputLen, 4)),
                                  static_cast<uint32_t>(kvNumHead * batch)};
        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_rearrangeV->get().setArg(index++, mGlobalWorkSizeRearrgV[0]);
        ret |= mKernel_rearrangeV->get().setArg(index++, mGlobalWorkSizeRearrgV[1]);
        ret |= mKernel_rearrangeV->get().setArg(index++, mGlobalWorkSizeRearrgV[2]);
        ret |= mKernel_rearrangeV->get().setArg(index++, openCLBuffer(value));
        ret |= mKernel_rearrangeV->get().setArg(index++, valueBuffer);
        ret |= mKernel_rearrangeV->get().setArg(index++, mPastKvSeqlen);
        ret |= mKernel_rearrangeV->get().setArg(index++, mKeyValueMaxlen);
        ret |= mKernel_rearrangeV->get().setArg(index++, kvInputLen);
        ret |= mKernel_rearrangeV->get().setArg(index++, kvNumHead);
        ret |= mKernel_rearrangeV->get().setArg(index++, headDim);
        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_v");
        mLocalWorkSizeRearrgV =
            localWS3DDefault(mGlobalWorkSizeRearrgV, maxWorkGroupSize, runtime, "rearrange_v", mKernel_rearrangeV,
                             mOpenCLBackend->getCLTuneLevel(), "attention_kv_rearrange_buf",
                             rearrangeVLwsShortlist(runtime->getGpuType()))
                .first;
        mGlobalWorkSizeRearrgV[0] =
            ROUND_UP(mGlobalWorkSizeRearrgV[0], std::max((uint32_t)1, mLocalWorkSizeRearrgV[0]));
        mGlobalWorkSizeRearrgV[1] =
            ROUND_UP(mGlobalWorkSizeRearrgV[1], std::max((uint32_t)1, mLocalWorkSizeRearrgV[1]));
        mGlobalWorkSizeRearrgV[2] =
            ROUND_UP(mGlobalWorkSizeRearrgV[2], std::max((uint32_t)1, mLocalWorkSizeRearrgV[2]));
        if (nullptr != mMeta) {
            mRgVUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &(*(mKVCacheCLManager->value()))()});
        }
        mRgVUpdateInfo.update_kernel_args.push_back({0, 5, sizeof(mPastKvSeqlen), &mPastKvSeqlen});
        mRgVUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
        mOpRecordUpdateInfo.emplace_back(&mRgVUpdateInfo);
        mOpenCLBackend->recordKernel3d(mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV,
                                       &mRgVUpdateInfo);
    }
    {
        // fused qk + mask + softmax + qkv
        mKernel_fa = runtime->buildKernel("attention_flash_prefill_buf", "flash_attention_prefill",
                                          flashPrefillBuildOptions(headDim, groupSize), mOpenCLBackend->getPrecision(),
                                          inputs[0], outputs[0]);
        mGlobalWorkSizeFa = {static_cast<uint32_t>(mFaWgSize * UP_DIV(seqlen, mFaTileQ)),
                             static_cast<uint32_t>(numHead * batch)};
        mLocalWorkSizeFa = {static_cast<uint32_t>(mFaWgSize), 1};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_fa->get().setArg(index++, openCLBuffer(query));
        ret |= mKernel_fa->get().setArg(index++, keyBuffer);
        ret |= mKernel_fa->get().setArg(index++, valueBuffer);
        ret |= mKernel_fa->get().setArg(index++, mHasMask ? openCLBuffer(inputs[3]) : openCLBuffer(query));
        ret |= mKernel_fa->get().setArg(index++, openCLBuffer(outputs[0]));
        ret |= mKernel_fa->get().setArg(index++, scale);
        ret |= mKernel_fa->get().setArg(index++, seqlen);
        ret |= mKernel_fa->get().setArg(index++, mKvSeqlen);
        ret |= mKernel_fa->get().setArg(index++, maskKvlen);
        ret |= mKernel_fa->get().setArg(index++, mKeyValueMaxlen);
        ret |= mKernel_fa->get().setArg(index++, numHead);
        ret |= mKernel_fa->get().setArg(index++, kvNumHead);
        ret |= mKernel_fa->get().setArg(index++, batch);
        ret |= mKernel_fa->get().setArg(index++, maskQlen);
        ret |= mKernel_fa->get().setArg(index++, maskBatchStride);
        MNN_CHECK_CL_SUCCESS(ret, "setArg flash_attention_prefill");

        if (nullptr != mMeta) {
            mFaUpdateInfo.update_kernel_args.push_back({0, 1, sizeof(cl_mem), &(*(mKVCacheCLManager->key()))()});
            mFaUpdateInfo.update_kernel_args.push_back({0, 2, sizeof(cl_mem), &(*(mKVCacheCLManager->value()))()});
        }
        mFaUpdateInfo.update_kernel_args.push_back({0, 7, sizeof(mKvSeqlen), &mKvSeqlen});
        mFaUpdateInfo.update_kernel_args.push_back({0, 9, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
        mOpRecordUpdateInfo.emplace_back(&mFaUpdateInfo);
        mOpenCLBackend->recordKernel2d(mKernel_fa, mGlobalWorkSizeFa, mLocalWorkSizeFa, &mFaUpdateInfo);
    }
    mOpenCLBackend->endRecord(mRecording);

    return NO_ERROR;
}

// [Batch, q_seqlen, HeadNum, HeadDim] -> [Batch, kv_seqlen, HeadNum, HeadDim]
// Flash-decoding: a split-kv partial pass plus a reduce replaces matmul_qk + softmax + matmul_qkv,
// and the partial appends this step's kv row itself, so rearrange_k / rearrange_v are prefill-only
// now and decode is down to two launches. The three-stage path runs its
// P*V matmul on ceil(headDim/8) * headNum work-items -- 256 for headDim 128 with 16 heads
// -- while streaming the whole V cache, so it leaves most of the GPU idle at long kv.
ErrorCode AttentionBufExecution::flashDecodeResize(const std::vector<Tensor*>& inputs,
                                                   const std::vector<Tensor*>& outputs) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto query = inputs[0];
    auto shape = query->shape();

    int numHead = shape[2];
    int kvNumHead = inputs[1]->shape()[2];
    int headDim = shape[3];
    int group_size = numHead / kvNumHead;
    float scale = (mAttnScale == 0.0f) ? (1.0f / sqrt(headDim)) : mAttnScale;

    // Without a KVMeta the cache manager never allocated anything (handleKVCache returns early),
    // so a caller that skips the KVCACHE_INFO hint needs scratch of its own -- same shape as the
    // three-stage path uses. kv length is then just this step's input, so nothing here has to be
    // re-patched later either.
    cl::Buffer keyBuffer, valueBuffer;
    if (nullptr != mMeta) {
        keyBuffer = *mKVCacheCLManager->key();
        valueBuffer = *mKVCacheCLManager->value();
    } else {
        int seqlen = shape[1];
        int batch = shape[0];
        mTempK.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, 4) * ROUND_UP(headDim, 4) * numHead * batch}));
        mTempV.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, 4) * ROUND_UP(headDim, 4) * numHead * batch}));
        mOpenCLBackend->onAcquireBuffer(mTempK.get(), Backend::DYNAMIC);
        mOpenCLBackend->onAcquireBuffer(mTempV.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mTempV.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mTempK.get(), Backend::DYNAMIC);
        keyBuffer = openCLBuffer(mTempK.get());
        valueBuffer = openCLBuffer(mTempV.get());
    }

    // The group size follows the wave, and the kv chunk is its own choice: both phases of the
    // partial kernel walk their axis grid-stride. headDim=128 keeps the (64, 64) pair it derived
    // before, when one value had to serve as both.
    mFdChunk = kFdChunk;
    mFdWgSize = std::min(kFdChunk, (int)mMaxWorkGroupSize);
    mFdNumChunk = UP_DIV(mKvSeqlen, mFdChunk);
    mFdMaxChunk = UP_DIV(mDecodeTmpMaxlen, mFdChunk);
    // int32 rather than float: the backend stores float tensors as half in low precision,
    // and the partials want the wider accumulator.
    mTempPartialO.reset(Tensor::createDevice<int32_t>({numHead * mFdMaxChunk * headDim}));
    mTempPartialML.reset(Tensor::createDevice<int32_t>({numHead * mFdMaxChunk * 2}));
    mOpenCLBackend->onAcquireBuffer(mTempPartialO.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onAcquireBuffer(mTempPartialML.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onReleaseBuffer(mTempPartialO.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onReleaseBuffer(mTempPartialML.get(), Backend::DYNAMIC_IN_EXECUTION);

    // No rearrange_k / rearrange_v here: the partial kernel appends this step's kv row to the cache
    // itself, in the one workgroup that reads it. See the comment there for why that is safe.
    {
        // split-kv partial pass
        std::set<std::string> buildOption;
        buildOption.emplace("-DNUMHEAD_GROUP_SIZE=" + std::to_string(group_size));
        buildOption.emplace("-DFD_HEAD_DIM=" + std::to_string(headDim));
        buildOption.emplace("-DFD_WG_SIZE=" + std::to_string(mFdWgSize));
        buildOption.emplace("-DFD_CHUNK=" + std::to_string(mFdChunk));
        if (runtime->getGpuType() == MALI) {
            buildOption.emplace("-DFD_QK_PAIR=1");
        }
        // The single-chunk path writes the output directly, so partial needs the same layout switch
        // reduce has.
        if (mOutputC4) {
            buildOption.emplace("-DATTENTION_C4");
        }
        mKernel_fdPartial = runtime->buildKernel("attention_flash_decode_partial_buf", "flash_decode_partial",
                                                 buildOption, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        mGwsFdPartial = {static_cast<uint32_t>(mFdWgSize * mFdNumChunk), static_cast<uint32_t>(numHead)};
        mLwsFdPartial = {static_cast<uint32_t>(mFdWgSize), 1};
        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_fdPartial->get().setArg(index++, openCLBuffer(query));
        ret |= mKernel_fdPartial->get().setArg(index++, keyBuffer);
        ret |= mKernel_fdPartial->get().setArg(index++, valueBuffer);
        ret |= mKernel_fdPartial->get().setArg(index++, openCLDeferBuffer(mTempPartialO.get()));
        ret |= mKernel_fdPartial->get().setArg(index++, openCLDeferBuffer(mTempPartialML.get()));
        ret |= mKernel_fdPartial->get().setArg(index++, scale);
        ret |= mKernel_fdPartial->get().setArg(index++, mKvSeqlen);
        ret |= mKernel_fdPartial->get().setArg(index++, mKeyValueMaxlen);
        ret |= mKernel_fdPartial->get().setArg(index++, numHead);
        ret |= mKernel_fdPartial->get().setArg(index++, mFdNumChunk);
        ret |= mKernel_fdPartial->get().setArg(index++, openCLBuffer(outputs[0]));
        ret |= mKernel_fdPartial->get().setArg(index++, openCLBuffer(inputs[1]));
        ret |= mKernel_fdPartial->get().setArg(index++, openCLBuffer(inputs[2]));
        // Write aliases of args 1 / 2. Passing the cache twice keeps the read pointers const, which
        // the qk main loop needs; see the kernel comment.
        ret |= mKernel_fdPartial->get().setArg(index++, keyBuffer);
        ret |= mKernel_fdPartial->get().setArg(index++, valueBuffer);
        MNN_CHECK_CL_SUCCESS(ret, "setArg flash_decode_partial");
        if (nullptr != mMeta) {
            mFdPartialUpdateInfo.update_kernel_args.push_back({0, 1, sizeof(cl_mem), &(*(mKVCacheCLManager->key()))()});
            mFdPartialUpdateInfo.update_kernel_args.push_back(
                {0, 2, sizeof(cl_mem), &(*(mKVCacheCLManager->value()))()});
            mFdPartialUpdateInfo.update_kernel_args.push_back(
                {0, 3, sizeof(cl_mem), &openCLDeferBuffer(mTempPartialO.get())()});
            mFdPartialUpdateInfo.update_kernel_args.push_back(
                {0, 4, sizeof(cl_mem), &openCLDeferBuffer(mTempPartialML.get())()});
            mFdPartialUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(mKvSeqlen), &mKvSeqlen});
            mFdPartialUpdateInfo.update_kernel_args.push_back({0, 7, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
            mFdPartialUpdateInfo.update_kernel_args.push_back({0, 9, sizeof(mFdNumChunk), &mFdNumChunk});
            mFdPartialUpdateInfo.update_kernel_args.push_back(
                {0, 13, sizeof(cl_mem), &(*(mKVCacheCLManager->key()))()});
            mFdPartialUpdateInfo.update_kernel_args.push_back(
                {0, 14, sizeof(cl_mem), &(*(mKVCacheCLManager->value()))()});
        }
        mFdPartialGlobal_size[0] = mGwsFdPartial[0];
        mFdPartialGlobal_size[1] = mGwsFdPartial[1];
        mFdPartialUpdateInfo.update_global_size.push_back({0, mFdPartialGlobal_size});
        mOpRecordUpdateInfo.emplace_back(&mFdPartialUpdateInfo);
        mOpenCLBackend->recordKernel2d(mKernel_fdPartial, mGwsFdPartial, mLwsFdPartial, &mFdPartialUpdateInfo);
    }
    {
        // combine the per-chunk partials
        std::set<std::string> buildOption;
        buildOption.emplace("-DFD_HEAD_DIM=" + std::to_string(headDim));
        buildOption.emplace("-DFD_WG_SIZE=" + std::to_string(mFdWgSize));
        if (mOutputC4) {
            buildOption.emplace("-DATTENTION_C4");
        }
        mKernel_fdReduce = runtime->buildKernel("attention_flash_decode_reduce_buf", "flash_decode_reduce", buildOption,
                                                mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_fdReduce));
        mGwsFdReduce = {static_cast<uint32_t>(UP_DIV(headDim, 4)), static_cast<uint32_t>(numHead)};
        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_fdReduce->get().setArg(index++, openCLDeferBuffer(mTempPartialO.get()));
        ret |= mKernel_fdReduce->get().setArg(index++, openCLDeferBuffer(mTempPartialML.get()));
        ret |= mKernel_fdReduce->get().setArg(index++, openCLBuffer(outputs[0]));
        ret |= mKernel_fdReduce->get().setArg(index++, numHead);
        ret |= mKernel_fdReduce->get().setArg(index++, mFdNumChunk);
        MNN_CHECK_CL_SUCCESS(ret, "setArg flash_decode_reduce");
        mLwsFdReduce =
            localWS2DDefault(mGwsFdReduce, maxWorkGroupSize, runtime, "flash_decode_reduce", mKernel_fdReduce,
                             mOpenCLBackend->getCLTuneLevel(), "attention_flash_decode_reduce_buf")
                .first;
        mGwsFdReduce[0] = ROUND_UP(mGwsFdReduce[0], std::max((uint32_t)1, mLwsFdReduce[0]));
        mGwsFdReduce[1] = ROUND_UP(mGwsFdReduce[1], std::max((uint32_t)1, mLwsFdReduce[1]));
        mFdReduceUpdateInfo.update_kernel_args.push_back(
            {0, 0, sizeof(cl_mem), &openCLDeferBuffer(mTempPartialO.get())()});
        mFdReduceUpdateInfo.update_kernel_args.push_back(
            {0, 1, sizeof(cl_mem), &openCLDeferBuffer(mTempPartialML.get())()});
        mFdReduceUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(mFdNumChunk), &mFdNumChunk});
        mOpRecordUpdateInfo.emplace_back(&mFdReduceUpdateInfo);
        mOpenCLBackend->recordKernel2d(mKernel_fdReduce, mGwsFdReduce, mLwsFdReduce, &mFdReduceUpdateInfo);
    }
    mOpenCLBackend->endRecord(mRecording);
    return NO_ERROR;
}

ErrorCode AttentionBufExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mOpenCLBackend->startRecord(mRecording);
    auto shape = inputs[0]->shape();

    int batch = shape[0];
    int seqlen = shape[1];
    int numHead = shape[2];
    int headDim = shape[3];
    int kvNumHead = inputs[1]->shape()[2];
    mHasMask = inputs.size() > 3 && inputs[3]->dimensions() > 2;
    mIsAddMask = mHasMask && inputs[3]->getType() == halide_type_of<float>();
    // Decode kernels have no batch axis, so batched single-token inputs use the prefill path.
    mIsDecode = seqlen == 1 && batch == 1 && (nullptr == mMeta || mMeta->add == 1);

    init();
    handleKVCache(inputs, outputs);

    mLongPrefill = false;
    mFlashPrefill = false;
    if (mIsDecode) {
        return flashDecodeResize(inputs, outputs);
    } else {
        const bool flashOk = flashPrefillEligible(inputs, outputs);
        // Long prefill treats the mask as a full additive GEMM bias and only lays out incoming KV,
        // so it requires an exact float mask plane and an empty cache.
        bool longOk = mHasMask && mIsAddMask && 0 == mPastKvSeqlen;
        if (longOk) {
            auto mask = inputs[3];
            int dim = mask->dimensions();
            longOk = mask->shape()[dim - 2] == seqlen && mask->shape()[dim - 1] == mKvSeqlen;
        }
        if (!flashOk && !longOk) {
            // flash now declines on one thing only: a device whose local memory or work-group
            // limit rules out every tiling in the search. Mask shape no longer keeps it out.
            MNN_ERROR("OpenCL attention: no prefill path for this shape (headDim=%d seqLen=%d mask=%s)\n", headDim,
                      seqlen, mHasMask ? (mIsAddMask ? "float plane" : "int plane") : "sentinel");
            return NOT_SUPPORT;
        }
        uint32_t mode = flashOk ? kPrefillFlash : kPrefillLong;
        if (mPastKvSeqlen == 0) {
            std::pair<std::vector<uint32_t>, uint32_t> tuneInfo;
            std::string info = "attention_" + std::to_string(batch) + "_" + std::to_string(numHead) + "_" +
                               std::to_string(headDim) + "_" + std::to_string(kvNumHead);
            if (seqlen > 16) {
                if (getTunedInfo(info, {static_cast<unsigned int>(seqlen)}, tuneInfo,
                                 mOpenCLBackend->getOpenCLRuntime(), mOpenCLBackend->getCLTuneLevel())) {
                    mode = tuneInfo.first[0];
                    if ((kPrefillFlash != mode && kPrefillLong != mode) || (kPrefillFlash == mode && !flashOk) ||
                        (kPrefillLong == mode && !longOk)) {
                        mode = flashOk ? kPrefillFlash : kPrefillLong;
                    }
                } else if (flashOk && longOk) {
                    // Measure only when both implementations satisfy the input constraints.
                    if (mOpenCLBackend->getCLTuneLevel() != None) {
                        setRecordClose closeRecord(mOpenCLBackend);
                        mLongPrefill = true;
                        longPrefillResize(inputs, outputs);
                        auto bestTime = measureExecuteTime();
                        mLongPrefill = false;
                        mode = kPrefillLong;
                        init();
                        mFlashPrefill = true;
                        // Can decline on the work-group probe; then it is simply not a rival.
                        if (NO_ERROR == flashPrefillResize(inputs, outputs)) {
                            auto flashPrefillTime = measureExecuteTime();
                            if (flashPrefillTime < bestTime) {
                                mode = kPrefillFlash;
                            }
                        }
                        mFlashPrefill = false;
                        std::pair<std::vector<uint32_t>, uint32_t> tuneInfoTmp =
                            std::make_pair<std::vector<uint32_t>, uint32_t>({mode}, 0);
                        setTunedInfo(
                            info, {static_cast<unsigned int>(seqlen)}, tuneInfoTmp, mOpenCLBackend->getOpenCLRuntime(),
                            "attention_long_layout_buf;attention_kv_rearrange_buf;attention_flash_prefill_buf");
                        init();
                    } else if (seqlen > 512) {
                        // Tuning is off, so fall back to the branch the measurement lands on for
                        // all but the shortest sequences.
                        mode = kPrefillLong;
                    }
                }
            }
        }
        mLongPrefill = (kPrefillLong == mode);
        mFlashPrefill = (kPrefillFlash == mode);
        if (mFlashPrefill) {
            // The only way this fails is the work-group probe at the top of it, which happens
            // before any state is built up. Long can take over when the mask lets it.
            if (NO_ERROR != flashPrefillResize(inputs, outputs)) {
                if (!longOk) {
                    MNN_ERROR("OpenCL attention: flash prefill found no runnable work group\n");
                    return NOT_SUPPORT;
                }
                mFlashPrefill = false;
                mLongPrefill = true;
                init();
                longPrefillResize(inputs, outputs);
            }
        } else {
            longPrefillResize(inputs, outputs);
        }
    }

    return NO_ERROR;
}

int AttentionBufExecution::getExecuteTime() {
    int executeTime = 0;
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (mFlashPrefill) {
        cl::Event event0, event1, event2;
        run3DKernelDefault(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg, runtime, &event0);
        executeTime += runtime->getEventTime(event0);
        run3DKernelDefault(mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV, runtime, &event1);
        executeTime += runtime->getEventTime(event1);
        runKernel2D(mKernel_fa, mGlobalWorkSizeFa, mLocalWorkSizeFa, runtime, &event2);
        executeTime += runtime->getEventTime(event2);
        return executeTime;
    }
    // Long prefill, the only other candidate the tuner ever measures.
    {
        int seq_idx = 0;
        cl::Event event0, event1, event2, event3, event4, event5, event6;
        run3DKernelDefault(mKernel_rearrange_vec[seq_idx], mGwsRearrgVec[seq_idx], mLwsRearrgVec[seq_idx],
                           mOpenCLBackend->getOpenCLRuntime(), &event0);
        executeTime += runtime->getEventTime(event0);
        if (mHasMask) {
            run3DKernelDefault(mKernel_mask_vec[seq_idx], mGwsMaskVec[seq_idx], mLwsMaskVec[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime(), &event1);
            executeTime += runtime->getEventTime(event1);
        }
        for (int seq_idx = 0; seq_idx < mQseqSplitNum; seq_idx++) {
            run3DKernelDefault(mKernel_qk_vec[seq_idx], mGwsQkVec[seq_idx], mLwsQkVec[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime(), &event2);
            executeTime += runtime->getEventTime(event2);
            run3DKernelDefault(mKernel_softmax_vec[seq_idx], mGwsSoftMaxVec[seq_idx], mLwsSoftMaxVec[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime(), &event3);
            executeTime += runtime->getEventTime(event3);
            run3DKernelDefault(mKernel_trans_vec[seq_idx], mGwsTransVec[seq_idx], mLwsTransVec[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime(), &event4);
            executeTime += runtime->getEventTime(event4);
            run3DKernelDefault(mKernel_qkv_vec[seq_idx], mGwsQkvVec[seq_idx], mLwsQkvVec[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime(), &event5);
            executeTime += runtime->getEventTime(event5);
        }
        seq_idx = 0;
        run3DKernelDefault(mKernel_clip_vec[seq_idx], mGwsClipVec[seq_idx], mLwsClipVec[seq_idx],
                           mOpenCLBackend->getOpenCLRuntime(), &event6);
        executeTime += runtime->getEventTime(event6);
    }
    return executeTime;
}

int AttentionBufExecution::measureExecuteTime() {
    return getExecuteTime();
}

ErrorCode AttentionBufExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start AttentionBufExecution onExecute !\n");
#endif
    if (nullptr != mMeta) {
        // If allocKVCache already ran reallocKVCache(true) during resize phase,
        // skip it here to avoid double-executing Remove. For subsequent decode
        // iterations (no resize), mReallocDone is false so we still run it.
        if (mKVCacheCLManager->isReallocDone()) {
            mKVCacheCLManager->clearReallocDone();
        } else {
            int kvInputLen = inputs[1]->shape()[1];
            mKVCacheCLManager->reallocKVCache(mMeta, kvInputLen);
        }
    }
    UpdateArgs(inputs, outputs);
#ifdef ENABLE_OPENCL_TIME_PROFILER
    if (mFlashPrefill) {
        cl::Event event0, event1, event2;
        run3DKernelDefault(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg,
                           mOpenCLBackend->getOpenCLRuntime(), &event0);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_k", event0});
        run3DKernelDefault(mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV,
                           mOpenCLBackend->getOpenCLRuntime(), &event1);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_v", event1});
        runKernel2D(mKernel_fa, mGlobalWorkSizeFa, mLocalWorkSizeFa, mOpenCLBackend->getOpenCLRuntime(), &event2);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"flash_attention_prefill", event2});
    } else if (mLongPrefill) {
        int seq_idx = 0;
        cl::Event event0, event1, event2, event3, event4, event5, event6;
        run3DKernelDefault(mKernel_rearrange_vec[seq_idx], mGwsRearrgVec[seq_idx], mLwsRearrgVec[seq_idx],
                           mOpenCLBackend->getOpenCLRuntime(), &event0);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_qkv", event0});
        if (mHasMask) {
            run3DKernelDefault(mKernel_mask_vec[seq_idx], mGwsMaskVec[seq_idx], mLwsMaskVec[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime(), &event1);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_mask", event1});
        }
        for (int seq_idx = 0; seq_idx < mQseqSplitNum; seq_idx++) {
            run3DKernelDefault(mKernel_qk_vec[seq_idx], mGwsQkVec[seq_idx], mLwsQkVec[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime(), &event2);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"matmul_qk_div_mask", event2});
            run3DKernelDefault(mKernel_softmax_vec[seq_idx], mGwsSoftMaxVec[seq_idx], mLwsSoftMaxVec[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime(), &event3);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"softmax", event3});
            run3DKernelDefault(mKernel_trans_vec[seq_idx], mGwsTransVec[seq_idx], mLwsTransVec[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime(), &event4);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"transpose_softmax", event4});
            run3DKernelDefault(mKernel_qkv_vec[seq_idx], mGwsQkvVec[seq_idx], mLwsQkvVec[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime(), &event5);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"matmul_qkv", event5});
        }
        seq_idx = 0;
        run3DKernelDefault(mKernel_clip_vec[seq_idx], mGwsClipVec[seq_idx], mLwsClipVec[seq_idx],
                           mOpenCLBackend->getOpenCLRuntime(), &event6);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_output", event6});
    } else {
        // Decode -- flash decoding, the only remaining non-prefill path. The kv rearrange is folded
        // into the partial kernel, so there is nothing to launch before it.
        cl::Event e2, e3;
        runKernel2D(mKernel_fdPartial, mGwsFdPartial, mLwsFdPartial, mOpenCLBackend->getOpenCLRuntime(), &e2);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"flash_decode_partial", e2});
        runKernel2D(mKernel_fdReduce, mGwsFdReduce, mLwsFdReduce, mOpenCLBackend->getOpenCLRuntime(), &e3);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"flash_decode_reduce", e3});
    }
#else
    if (mOpenCLBackend->isUseRecordQueue()) {
        mOpenCLBackend->addRecord(mRecording, mOpRecordUpdateInfo);
#ifdef LOG_VERBOSE
        MNN_PRINT("End AttentionBufExecution onExecute... \n");
#endif
        return NO_ERROR;
    }

    if (mFlashPrefill) {
        run3DKernelDefault(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg,
                           mOpenCLBackend->getOpenCLRuntime());
        run3DKernelDefault(mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV,
                           mOpenCLBackend->getOpenCLRuntime());
        runKernel2D(mKernel_fa, mGlobalWorkSizeFa, mLocalWorkSizeFa, mOpenCLBackend->getOpenCLRuntime());
    } else if (mLongPrefill) {
        int seq_idx = 0;
        run3DKernelDefault(mKernel_rearrange_vec[seq_idx], mGwsRearrgVec[seq_idx], mLwsRearrgVec[seq_idx],
                           mOpenCLBackend->getOpenCLRuntime());
        if (mHasMask) {
            run3DKernelDefault(mKernel_mask_vec[seq_idx], mGwsMaskVec[seq_idx], mLwsMaskVec[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime());
        }
        for (int seq_idx = 0; seq_idx < mQseqSplitNum; seq_idx++) {
            run3DKernelDefault(mKernel_qk_vec[seq_idx], mGwsQkVec[seq_idx], mLwsQkVec[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime());
            run3DKernelDefault(mKernel_softmax_vec[seq_idx], mGwsSoftMaxVec[seq_idx], mLwsSoftMaxVec[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime());
            run3DKernelDefault(mKernel_trans_vec[seq_idx], mGwsTransVec[seq_idx], mLwsTransVec[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime());
            run3DKernelDefault(mKernel_qkv_vec[seq_idx], mGwsQkvVec[seq_idx], mLwsQkvVec[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime());
        }
        seq_idx = 0;
        run3DKernelDefault(mKernel_clip_vec[seq_idx], mGwsClipVec[seq_idx], mLwsClipVec[seq_idx],
                           mOpenCLBackend->getOpenCLRuntime());
    } else {
        // Decode -- flash decoding, the only remaining non-prefill path. The kv rearrange is folded
        // into the partial kernel, so there is nothing to launch before it.
        runKernel2D(mKernel_fdPartial, mGwsFdPartial, mLwsFdPartial, mOpenCLBackend->getOpenCLRuntime());
        runKernel2D(mKernel_fdReduce, mGwsFdReduce, mLwsFdReduce, mOpenCLBackend->getOpenCLRuntime());
    }
#endif

    // Prefill-only prefix cache dump: persist this layer's prompt kvcache to disk once the
    // rearrange kernels have populated the buffer (savePrefixKVCache finish()es the queue).
    // Safe against the record queue: the LLM engine disables it during prefill (sets
    // OP_ENCODER_NUMBER_FOR_COMMIT=0), so the write path always reaches here via synchronous
    // dispatch rather than the addRecord early-return above.
    if (!mIsDecode && nullptr != mMeta && mMeta->file_flag == KVMeta::PendingWrite &&
        mKVCacheCLManager->savingPrefix()) {
        mKVCacheCLManager->savePrefixKVCache();
    }

#ifdef LOG_VERBOSE
    MNN_PRINT("end AttentionBufExecution onExecute !\n");
#endif

    return NO_ERROR;
}

AttentionBufExecution::AttentionBufExecution(const MNN::Op* op, Backend* backend, bool outputC4)
    : CommonExecution(backend, op) {
    mMeta = (KVMeta*)(backend->getMetaPtr());
    mOutputC4 = outputC4;
    mAttnScale = op->main_as_AttentionParam()->attnScale();
    mKVCacheCLManager.reset(new KVCacheCLManager(backend, nullptr != mMeta));
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    mKVCacheCLManager->setPrefixCacheDir(mOpenCLBackend->getRuntime()->hint().prefixcacheDirPath);
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel(
        "softmax_buf", "softmax_buf", {"-DSOFTMAX_LOCAL_SIZE=512"}, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL_CTOR(kernel);
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
}

AttentionBufExecution::AttentionBufExecution(std::shared_ptr<KVCacheCLManager> manager, const MNN::Op* op,
                                             Backend* backend)
    : CommonExecution(backend, op), mKVCacheCLManager(manager) {
    mMeta = (KVMeta*)(backend->getMetaPtr());
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    auto param = op->main_as_AttentionParam();
    mOutputC4 = param->output_c4();
    mAttnScale = param->attnScale();
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel(
        "softmax_buf", "softmax_buf", {"-DSOFTMAX_LOCAL_SIZE=512"}, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL_CTOR(kernel);
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
}

void AttentionBufExecution::prebuildOpenCLPrograms(const std::vector<Tensor*>& inputs,
                                                   const std::vector<Tensor*>& outputs) {
    if (inputs.size() < 3 || outputs.empty() || inputs[0]->dimensions() < 4 || inputs[1]->dimensions() < 3) {
        return;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const int precision = mOpenCLBackend->getPrecision();
    const int numHead = inputs[0]->length(2);
    const int kvNumHead = inputs[1]->length(2);
    const int headDim = inputs[0]->length(3);
    if (numHead <= 0 || kvNumHead <= 0 || headDim <= 0 || numHead % kvNumHead != 0) {
        return;
    }
    const int groupSize = numHead / kvNumHead;

    runtime->submitPrebuild("attention_long_layout_buf", {}, precision, inputs[0], outputs[0], true);
    runtime->submitPrebuild("attention_kv_rearrange_buf", {}, precision, inputs[0], outputs[0], true);
    if (TensorUtils::getDescribe(inputs[2])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        runtime->submitPrebuild("attention_long_layout_buf", {"-DVALUE_C4"}, precision, inputs[0], outputs[0], true);
        runtime->submitPrebuild("attention_kv_rearrange_buf", {"-DVALUE_C4"}, precision, inputs[0], outputs[0], true);
    }

    mHasMask = inputs.size() > 3 && inputs[3]->dimensions() > 2;
    mIsAddMask = mHasMask && inputs[3]->getType() == halide_type_of<float>();
    if (flashPrefillEligible(inputs, outputs)) {
        runtime->submitPrebuild("attention_flash_prefill_buf", flashPrefillBuildOptions(headDim, groupSize), precision,
                                inputs[0], outputs[0], true);
    }

    const int fdWgSize = std::min(kFdChunk, static_cast<int>(mMaxWorkGroupSize));
    std::set<std::string> partialOptions = {
        "-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize),
        "-DFD_HEAD_DIM=" + std::to_string(headDim),
        "-DFD_WG_SIZE=" + std::to_string(fdWgSize),
        "-DFD_CHUNK=" + std::to_string(kFdChunk),
    };
    if (runtime->getGpuType() == MALI) {
        partialOptions.emplace("-DFD_QK_PAIR=1");
    }
    if (mOutputC4) {
        partialOptions.emplace("-DATTENTION_C4");
    }
    runtime->submitPrebuild("attention_flash_decode_partial_buf", partialOptions, precision, inputs[0], outputs[0],
                            true);

    std::set<std::string> reduceOptions = {"-DFD_HEAD_DIM=" + std::to_string(headDim),
                                           "-DFD_WG_SIZE=" + std::to_string(fdWgSize)};
    if (mOutputC4) {
        reduceOptions.emplace("-DATTENTION_C4");
    }
    runtime->submitPrebuild("attention_flash_decode_reduce_buf", reduceOptions, precision, inputs[0], outputs[0], true);
}

bool AttentionBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    if (bn->getMetaPtr() == mMeta && mMeta != nullptr) {
        *dst = new AttentionBufExecution(mKVCacheCLManager, op, bn);
    } else {
        *dst = new AttentionBufExecution(op, bn, op->main_as_AttentionParam()->output_c4());
    }
    return true;
}

class AttentionBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        auto param = op->main_as_AttentionParam();
        OPENCL_CREATOR_CHECK(new AttentionBufExecution(op, backend, param->output_c4()));
    }
};
REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(AttentionBufCreator, OpType_Attention, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */