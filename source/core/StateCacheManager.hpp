//  StateCacheManager.hpp
//  Paged Attention Algorithm Implementation
//
//  Created by [Your Name] on [Creation Date].
//  Copyright © [Year], [Your Organization]
//

#ifndef StateCacheManager_hpp
#define StateCacheManager_hpp

#include <vector>
#include <list>
#include <queue>
#include <unordered_map>
#include <memory>
#include <cassert>
#include <iostream>
#include <functional> 

#include <MNN/Tensor.hpp>

namespace MNN {

enum class MNNStateCacheType {
    MNN_STATECACHE_NAIVE = 0, // with only pre-allocation
    MNN_STATECACHE_ADVANCED = 1, // with pre-allocation, paging, offloading.
};

enum class MNNStateCacheQuantType {
    NoQuant = 0, // 0: Do not quantize kvcache, just store float
    QuantKeyInt8 = 1, // 1: Only quantize key cache, use int8 asymmetric quantization 
    QuantValueFp8 = 2, // 2: Only quantize value cache, use fp8 quantization
    QuantKeyInt8ValueFp8 = 3, // 3: quantize both key and value cache as described above
    QuantValueInt8 = 4, // 4: Only quantize value cache, use int8 asymmetric quantization 
    QuantKeyInt8ValueInt8 = 5, // 5: quantize both key and value cache as described above
};

/* 2.1 StateCacheBlock 
    All the blocks are of the same size.
    */ 
class MNN_PUBLIC StateCacheBlock {
private:
    std::vector<int> mRefIds; // IDs of samples using this block
    int mSlotNum; // Index pointing to the id of the next available slot in this block
    std::vector<Tensor*> mTensors; // Tensors holding the KV cache data
    std::vector<int> mTensorSize;
    int mBlockSize;
public:
    struct LAYOUT {
        enum class NoQuant {
            PAST_K = 0,
            PAST_V = 1
        };
        enum class QuantKeyInt8 {
            PAST_K = 0,
            PAST_K_SCALES = 1,
            PAST_K_ZERO_POINTS = 2,
            PAST_V = 3
        };
        enum class QuantValueFp8 {
            PAST_K = 0,
            PAST_K_SCALES = 1,
            PAST_K_ZERO_POINTS = 2,
            PAST_V = 3
        };
        enum class QuantKeyInt8ValueFp8 {
            PAST_K = 0,
            PAST_K_SCALES = 1,
            PAST_K_ZERO_POINTS = 2,
            PAST_V = 3
        };
        enum class QuantValueInt8 {
            PAST_K = 0,
            PAST_V = 1,
            PAST_V_SCALES = 2,
            PAST_V_ZERO_POINTS = 3
        };
        enum class QuantKeyInt8ValueInt8 {
            PAST_K = 0,
            PAST_K_SCALES = 1,
            PAST_K_ZERO_POINTS = 2,
            PAST_V = 3,
            PAST_V_SCALES = 4,
            PAST_V_ZERO_POINTS = 5
        };
    };
    StateCacheBlock(std::vector<int> ref_ids, int blok_size, int slot_num=0);
    // Tensor operations
    void setTensor(int tId, Tensor* tensor);
    void setTensorSize(int tId, int tensor_size);
    Tensor* getTensor(int tId) {
        return mTensors[tId];
    }
    // manage pointers and offsets
    bool onAllocatePtr(uint8_t* ptr);
    bool onAllocateOffset(size_t offset);
    // reset slot_num
    void resetSlotNum(int slot_num);
    int getSlotNum() {
        return mSlotNum;
    }
    // deal with sample reference
    void removeRef(int ref_id);
    void addRef(int ref_id);
    int refCount() const {
        return mRefIds.size();
    }
    bool needDesert() const {
        return refCount()==0;
    }
    // is full
    bool isFull() const {
        return mBlockSize==mSlotNum;
    }
};

// 2.2 StateCache
class MNN_PUBLIC StateCache {
public:
    // List of pointers to free memory blocks
    std::list<std::shared_ptr<uint8_t*>> freePtrList;

    // List of offsets in external storage for free blocks
    std::list<size_t> freeFileOffsetList;

    // Dynamic structure for in-memory blocks with minimal ref_ids size for eviction
    std::priority_queue<std::shared_ptr<StateCacheBlock>, std::vector<std::shared_ptr<StateCacheBlock>>, 
                        std::function<bool(const std::shared_ptr<StateCacheBlock>&, const std::shared_ptr<StateCacheBlock>&)>> 
        inMemBlockList {[](const std::shared_ptr<StateCacheBlock>& a, const std::shared_ptr<StateCacheBlock>& b) { return a->refCount() > b->refCount(); }};

    // Linked list of blocks currently being used for computation
    std::list<std::shared_ptr<StateCacheBlock>> computeCacheBlockList;

    // Linked list of blocks stored in external storage
    std::list<std::shared_ptr<StateCacheBlock>> offloadedCacheBlockList;
};

// 2.3 StateCacheManager
class MNN_PUBLIC StateCacheManager {
private:
    std::shared_ptr<StateCache> mStateCache;
    MNNStateCacheQuantType mQuantType;
    MNNStateCacheType mType;

public:
    StateCacheManager(MNNStateCacheQuantType quantType = MNNStateCacheQuantType::NoQuant, MNNStateCacheType type = MNNStateCacheType::MNN_STATECACHE_ADVANCED);
    // Enlarge the memory cache
    bool enlargeMemCache(size_t size);
    // Enlarge the file cache
    bool enlargeFileCache(size_t size);
    // Release the memory cache
    void releasesMemCache();
    // Release the file cache
    void releaseFileCache();
    // Evict a block
    std::shared_ptr<StateCacheBlock> evictBlock(const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list);
    // Get a free pointer
    std::shared_ptr<StateCacheBlock> getFreePtr(const std::vector<std::shared_ptr<StateCacheBlock>>& evict_pin_block_list);
    // Recover a block from the file
    void recoverBlock(std::shared_ptr<StateCacheBlock> block_ptr, const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list);
    // Desert a block
    void desertBlock(int ref_id, std::shared_ptr<StateCacheBlock> block_ptr);
    // Copy a block
    std::shared_ptr<StateCacheBlock> copyBlock(int ref_id, std::shared_ptr<StateCacheBlock> block_ptr);
    // Prepare attention
    void prepareAttn(int ref_id, const std::vector<std::shared_ptr<StateCacheBlock>>& argv);
};

} // namespace MNN

#endif // StateCacheManager_hpp