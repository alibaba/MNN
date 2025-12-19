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

namespace MNN {
    
void MetalKVCacheManager::onResize(int kv_num_head, int head_dim) {
    mKvNumHead = kv_num_head;
    mHeadDim = head_dim;
}

void MetalKVCacheManager::onAlloc(KVMeta* meta, int seq_len) {
    mMeta = meta;
    auto mtbn = static_cast<MetalBackend *>(mBackend);
    auto context = (__bridge MNNMetalContext *)mtbn->context();

    auto kv_seq_len = mMeta != nullptr ? mMeta->add : seq_len;
    int byte = 4;
    if(mtbn->useFp16InsteadFp32()) {
        byte = 2;
    }
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
        auto oldTotalSize = ALIMIN(oldKeySize, oldValueSize);
        if(oldKeySize != oldValueSize) {
            MNN_ERROR("[Error]: Kvcache in disk size of key and value should equal with metal backend\n");
        }
        size_t oldKeyMaxLength = oldKeySize / (mKvNumHead * mHeadDim * byte);
        size_t oldValueMaxLength = oldValueSize / (mKvNumHead * mHeadDim * byte);
        size_t oldMaxLength = ALIMIN(oldKeyMaxLength, oldValueMaxLength);
        if(oldMaxLength < meta->seqlen_in_disk) {
            MNN_ERROR("[Error]: Kvcache in disk size smaller than saved lengthInDiskToload:%d\n", (int)meta->seqlen_in_disk);
        }

        int kv_seq_len = ROUND_UP(meta->add + meta->seqlen_in_disk, mConfig.mKvAlignNum);
        mMaxLength = kv_seq_len > oldMaxLength ? ROUND_UP(meta->add + meta->seqlen_in_disk + mConfig.mExpandChunk, mConfig.mKvAlignNum) : oldMaxLength;
        size_t totalSize = mKvNumHead * mMaxLength * mHeadDim * byte;
        mCurrentTotalSize = totalSize;

        size_t old_piece_size = meta->seqlen_in_disk * byte;
        size_t old_piece_stride = oldMaxLength * byte;
        size_t new_piece_stride = mMaxLength * byte;
        
        mCurrentTotalSize = ALIMAX(mCurrentTotalSize, oldKeySize);
        mCurrentTotalSize = ALIMAX(mCurrentTotalSize, oldValueSize);

        createKVCacheFile();
        resetKVCacheFileSize(mCurrentTotalSize, mCurrentTotalSize);
        expandKVCacheInDisk(oldTotalSize, mCurrentTotalSize, old_piece_stride, old_piece_size, new_piece_stride, true, old_key_fd, old_value_fd);

        mPastLength = meta->seqlen_in_disk;
        mKVCacheInDisk = true;

        return;
    }
    
    // align max kv_seq_len to mKvAlignNum, for simd/tensor matrix load alignment
    mMaxLength = ROUND_UP(kv_seq_len + mConfig.mExpandChunk, mConfig.mKvAlignNum);
    size_t totalSize = mKvNumHead * mMaxLength * mHeadDim * byte;
    mCurrentTotalSize = totalSize;
    bool storeKvInDisk  = !mConfig.mKVCacheDir.empty();
    bool sharePrefixKv = mMeta != nullptr && mMeta->file_name.size() > 0 && mMeta->file_flag == KVMeta::PendingWrite;

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
        resetKVCacheFileSize(totalSize, totalSize);
        mmapKVCache(totalSize, totalSize);
        mKVCacheInDisk = true;
        
        mKeyBuffer   = [[context device] newBufferWithBytesNoCopy:mMapKeyAddr length:totalSize options:MTLResourceStorageModeShared  deallocator:nil];
        mValueBuffer = [[context device] newBufferWithBytesNoCopy:mMapValueAddr length:totalSize options:MTLResourceStorageModeShared  deallocator:nil];
        
        auto new_key_ptr = (uint8_t*)[mKeyBuffer contents];
        ::memset(new_key_ptr, 0, mMaxLength * mKvNumHead * mHeadDim * byte);
        
        auto new_value_ptr = (uint8_t*)[mValueBuffer contents];
        ::memset(new_value_ptr, 0, mMaxLength * mKvNumHead * mHeadDim * byte);

    } else {
        // past_key: [maxlen, kvNumhead, headdim]
        auto new_key = Tensor::createDevice<float>({mMaxLength, mKvNumHead, mHeadDim});
        // past_value: [kvNumhead, headdim, maxlen]
        auto new_value = Tensor::createDevice<float>({mKvNumHead, mHeadDim, mMaxLength});


        auto res = mBackend->onAcquireBuffer(new_key, Backend::STATIC);
        res = res && mBackend->onAcquireBuffer(new_value, Backend::STATIC);
        if(!res) {
            MNN_ERROR("attition kv cache alloc memory error:%d\n", res);
        }
        // memset for qkv matmul mad, in case dirty data
        auto newKeyBuf = MetalBackend::getBuffer(new_key);
        auto new_key_ptr = (uint8_t*)[newKeyBuf.first contents] + newKeyBuf.second;
        ::memset(new_key_ptr, 0, mMaxLength * mKvNumHead * mHeadDim * byte);
        
        auto newValueBuf = MetalBackend::getBuffer(new_value);
        auto new_value_ptr = (uint8_t*)[newValueBuf.first contents] + newValueBuf.second;
        ::memset(new_value_ptr, 0, mMaxLength * mKvNumHead * mHeadDim * byte);
        
        mPastKey.reset(new_key);
        mPastValue.reset(new_value);
    }
    
}
void MetalKVCacheManager::onRealloc(KVMeta* meta) {
    mMeta = meta;
    auto kv_seq_len = mMeta->previous + mMeta->add - mMeta->remove + mMeta->computeReverseSize();
    auto mtbn = static_cast<MetalBackend *>(mBackend);
    
    int byte = 4;
    if(mtbn->useFp16InsteadFp32()) {
        byte = 2;
    }
    
    auto start = mPastLength - mMeta->remove;
    // latest length larger than maxLen
    if (kv_seq_len > mMaxLength) {

        // copy mPastLength including all remove/reverse to new buffer first
        auto copy_len = mPastLength;
        bool needCopy = mPastLength > 0;
        
        size_t old_size = mKvNumHead * copy_len * mHeadDim * byte;
        size_t old_piece_size = copy_len * byte;
        size_t old_piece_stride = mMaxLength * byte;

        // align max kv_seq_len to mKvAlignNum, for simd/tensor matrix load alignment
        mMaxLength = ROUND_UP(kv_seq_len + mConfig.mExpandChunk, mConfig.mKvAlignNum);
        
        auto oldTotalSize = mCurrentTotalSize;
        size_t size = mKvNumHead * mMaxLength * mHeadDim * byte;
        mCurrentTotalSize = size;
        size_t new_piece_stride = mMaxLength * byte;
        
        mPastLength = (int)start;

        if(mKVCacheInDisk) {
            expandKVCacheInDisk(oldTotalSize, mCurrentTotalSize, old_piece_stride, old_piece_size, new_piece_stride, needCopy);
        } else {
            expandKVCacheInMem(oldTotalSize, old_piece_stride, old_piece_size, new_piece_stride, needCopy);
        }
    }
    
    // Remove
    {
        if (0 == mMeta->n_reserve) {
            mPastLength = start;
            return;
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
            // past_value : [mKvNumHead, mHeadDim, mCache->mMaxLength]

            auto copy_src_index = src_start + begin;
            auto copy_dst_index = start;
            for(int i = 0; i < length; i++) {
                ::memcpy(key_ptr + (copy_dst_index + i) * mKvNumHead * mHeadDim * byte, key_ptr + (copy_src_index + i) * mKvNumHead * mHeadDim * byte, mKvNumHead * mHeadDim * byte);
            }
            for(int j = 0; j <  mKvNumHead * mHeadDim; j++) {
                for(int i = 0; i < length; i++) {
                    ::memcpy(value_ptr + (j * mMaxLength + copy_dst_index + i) * byte, value_ptr + (j * mMaxLength + copy_src_index + i) * byte, byte);
                }
            }
            start += length;
        }
        mPastLength = (int)start;
    }
}
    
void MetalKVCacheManager::expandKVCacheInMem(size_t oldSize, size_t old_piece_stride, size_t old_piece_size, size_t new_piece_stride, bool need_copy) {
    auto mtbn = static_cast<MetalBackend *>(mBackend);
    int byte = 4;
    if(mtbn->useFp16InsteadFp32()) {
        byte = 2;
    }
    // past_key: [maxlen, kvNumhead, headdim]
    auto new_key = Tensor::createDevice<float>({mMaxLength, mKvNumHead, mHeadDim});
    // past_value: [kvNumhead, headdim, maxlen]
    auto new_value = Tensor::createDevice<float>({mKvNumHead, mHeadDim, mMaxLength});
    
    auto res = mBackend->onAcquireBuffer(new_key, Backend::STATIC);
    res = res && mBackend->onAcquireBuffer(new_value, Backend::STATIC);
    if(!res) {
        MNN_ERROR("attition kv cache realloc memory error:%d\n", res);
    }
    
    // memset for qkv matmul mad, in case dirty data
    auto newKeyBuf = MetalBackend::getBuffer(new_key);
    auto new_key_ptr = (uint8_t*)[newKeyBuf.first contents] + newKeyBuf.second;
    ::memset(new_key_ptr, 0, mMaxLength * mKvNumHead * mHeadDim * byte);
    
    auto newValueBuf = MetalBackend::getBuffer(new_value);
    auto new_value_ptr = (uint8_t*)[newValueBuf.first contents] + newValueBuf.second;
    ::memset(new_value_ptr, 0, mMaxLength * mKvNumHead * mHeadDim * byte);
    
    if (need_copy) {
        auto keyBuf = MetalBackend::getBuffer(mPastKey.get());
        auto key_ptr = (uint8_t*)[keyBuf.first contents] + keyBuf.second;;
        ::memcpy(new_key_ptr, key_ptr, oldSize);
        
        auto valueBuf = MetalBackend::getBuffer(mPastValue.get());
        auto value_ptr = (uint8_t*)[valueBuf.first contents] + valueBuf.second;
        for(int i = 0; i <  mKvNumHead * mHeadDim; i++) {
            ::memcpy(new_value_ptr + i * new_piece_stride, value_ptr + i * old_piece_stride, old_piece_size);
        }
    }
    
    mPastKey.reset(new_key);
    mPastValue.reset(new_value);
}
    
void MetalKVCacheManager::expandKVCacheInDisk(size_t oldSize, size_t curSize, size_t old_piece_stride, size_t old_piece_size, size_t new_piece_stride, bool need_copy, file_t specKeyFile, file_t specValueFile) {
    auto mtbn = static_cast<MetalBackend *>(mBackend);
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    
    mmapKVCache(oldSize, oldSize, specKeyFile, specValueFile);
    std::vector<int8_t> prevKey, prevValue;
    prevKey.resize(oldSize);
    prevValue.resize(oldSize);
    memcpy(prevKey.data(),   mMapKeyAddr,   oldSize);
    memcpy(prevValue.data(), mMapValueAddr, oldSize);
    
    unmapKVCache(oldSize, oldSize);
    resetKVCacheFileSize(curSize, curSize);
    mmapKVCache(curSize, curSize);
    
    // reset id<MTLBuffer>
    mKeyBuffer   = [[context device] newBufferWithBytesNoCopy:mMapKeyAddr length:curSize options:MTLResourceStorageModeShared  deallocator:nil];
    mValueBuffer = [[context device] newBufferWithBytesNoCopy:mMapValueAddr length:curSize options:MTLResourceStorageModeShared  deallocator:nil];
    
    
    // Step 3: Move the kvcache from temporary buffers in memory to disk
    memset(mMapKeyAddr, 0, curSize);
    memset(mMapValueAddr, 0, curSize);
    
    if (need_copy) {
        ::memcpy(mMapKeyAddr, prevKey.data(), oldSize);
        for(int i = 0; i <  mKvNumHead * mHeadDim; i++) {
            ::memcpy(mMapValueAddr + i * new_piece_stride, prevValue.data() + i * old_piece_stride, old_piece_size);
        }
    }
}
    
void MetalKVCacheManager::onClear() {
    if (mKVCacheInDisk) {
        mKeyBuffer = nil;
        mValueBuffer = nil;
        
        // mSaveShareKvPrefix also need unmap file
        unmapKVCache(mCurrentTotalSize, mCurrentTotalSize);
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
    mMaxLength = 0;
    mPastLength = 0;
}
} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

