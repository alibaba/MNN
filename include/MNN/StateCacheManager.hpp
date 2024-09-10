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
#include <set>
#include <map>
#include <unordered_map>
#include <memory>
#include <cassert>
#include <iostream>
#include <functional> 
#include <utility>

#include <MNN/Tensor.hpp>
#include <MNN/MNNForwardType.h>

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
    Tensor shape: 
        K: [num_heads, block_size / hP, head_dim, hP]
        V: [num_heads, head_dim / hP, block_size, hP]
    */ 
class MNN_PUBLIC StateCacheBlock {
private:
    std::set<int> mRefIds; // IDs of samples using this block
    // slot number is changed when numbers are actually filled in.
    int mSlotNum; // Index pointing to the id of the next available slot in this block
    std::vector<Tensor*> mTensors; // Tensors holding the KV cache data
    std::vector<int> mTensorSize;
    std::vector<int> mTensorBytes;
    std::vector<int> mTensorSeqLenDim;
    int mBlockSize;
    char* mBasePtr;
    size_t mSize;
    void initTensorInfo(int tensor_num);
    static std::pair<int, int> getCopyIter(StateCacheBlock* block, int index);
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
            PAST_V = 1
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
    StateCacheBlock(std::set<int> ref_ids, int blok_size, int slot_num=0);
    // Tensor operations
    // Tensor shape: K: [kvnum_heads, block_size / hP, head_dim, hP], V: [kvnum_heads, head_dim / hP, block_size, hP]
    // block_size % hP == 0, block_size % core->pack == 0 
    void setTensor(int tId, Tensor* tensor);
    void setTensor(int tId, Tensor* tensor, int bytes);
    void setTensorSize(int tId, int tensor_size);
    void resetTensorShape(std::vector<std::vector<int>>& shape, int hP);
    size_t setTensors(std::vector<std::vector<int>>& shape, void* backend, MNNStateCacheQuantType type, BackendConfig::PrecisionMode precision, int hP);
    Tensor* getTensor(int tId) {
        return mTensors[tId];
    }
    int getTensorSize(int tId) {
        return mTensorSize[tId];
    }
    int getTensorsLength() {
        return mTensors.size();
    }
    // manage pointers and offsets
    void onAllocatePtr(char* ptr);
    void onAllocateOffset(size_t offset);
    void setBlockMem(char* ptr, size_t size);
    char* getBlockPtr() const {
        return mBasePtr;
    }
    size_t getBlockPhysicalSize() const {
        return mSize;
    }
    // reset slot_num
    void resetSlotNum(int slot_num);
    int getSlotNum() {
        return mSlotNum;
    }
    bool isFull() const {
        return mBlockSize==mSlotNum;
    }
    int getFreeSlotNum() const {
        return mBlockSize - mSlotNum;
    }
    // deal with sample reference
    void removeRef(int ref_id);
    void clearRef();
    void addRef(int ref_id);
    int refCount() const {
        return mRefIds.size();
    }
    bool needDesert() const {
        return refCount()==0;
    }
    // is ownership
    void setRefIds(const std::set<int>& ref_ids){
        mRefIds = ref_ids;
    }
    std::set<int> getRefIds() const{
        return mRefIds;
    }
    bool own(int ref_id) const {
        return mRefIds.count(ref_id) != 0;
    }
     // Block size management
    void setBlockSize(int block_size){
        mBlockSize = block_size;
    }
    int getBlockSize() const{
        return mBlockSize;
    }
    // copy
    void copyBlock(std::shared_ptr<StateCacheBlock> src);
    // destructor
    ~StateCacheBlock();
};

// 2.2 StateCache
class MNN_PUBLIC StateCache {
public:
    // allocated pointer list, used for free()
    std::set<char*> mallocPtrList;
    // List of pointers to free memory blocks
    std::list<char*> freePtrList;
    // List of offsets in external storage for free blocks
    std::list<size_t> freeFileOffsetList;
    // Dynamic structure for in-memory blocks with minimal ref_ids size for eviction
    std::map<int, std::set<std::shared_ptr<StateCacheBlock>>> inMemBlockList;
    // Linked list of blocks currently being used for computation
    std::set<std::shared_ptr<StateCacheBlock>> computeCacheBlockList;
    // Linked list of blocks stored in external storage
    std::set<std::shared_ptr<StateCacheBlock>> offloadedBlockList;
public:
    void clear(std::shared_ptr<StateCacheBlock> block);
};

// 2.3 StateCacheReference
class MNN_PUBLIC StateCacheReference {
public:
    int mRefId;
    int mBlockSize;
    std::unordered_map<void*, std::vector<std::shared_ptr<StateCacheBlock>>> mPageTable;
    StateCacheReference(int refId, int blockSize) : mRefId(refId), mBlockSize(blockSize) {};
    StateCacheReference(int refId, std::shared_ptr<StateCacheReference> other) : mRefId(refId), mBlockSize(other->mBlockSize), mPageTable(other->mPageTable) {};
    int getSlotNum(void* layer) {
        if (mPageTable.count(layer)==0) return 0;
        int page_num = mPageTable[layer].size();
        return (page_num-1)*mPageTable[layer].back()->getBlockSize() + mPageTable[layer].back()->getSlotNum();
    }
    int getLogicalBlockId(int tokenId) {
        return tokenId / mBlockSize;
    }
    std::shared_ptr<StateCacheBlock> getPhysicalBlock(void* layer, int tokenId) {
        return mPageTable[layer][getLogicalBlockId(tokenId)];
    }
};

// 2.4 StateCacheManager
class MNN_PUBLIC StateCacheManager {
public:
struct StateCacheManagerConfig {
    int preallocateBlockNum = 8;
    int preallocateTokenNum = 63;
    int blockSize = 8;
};

private:
    std::unordered_map<void*, std::shared_ptr<StateCache>> mStateCache;
    MNNStateCacheQuantType mQuantType;
    MNNStateCacheType mType;
    BackendConfig::PrecisionMode FP_precision = BackendConfig::Precision_Normal;

    // Reference correlated stuff
    int mNextNewRefId;
    int mNextNewLayerId;
    int mBlockSize = 0;
    std::shared_ptr<StateCacheReference> mCurrentReference;

    // config
    struct StateCacheManagerConfig mConfig;

public:
    StateCacheManager(MNNStateCacheQuantType quantType = MNNStateCacheQuantType::NoQuant, MNNStateCacheType type = MNNStateCacheType::MNN_STATECACHE_ADVANCED);
    void setHint(MNNStateCacheQuantType quantType = MNNStateCacheQuantType::NoQuant, MNNStateCacheType type = MNNStateCacheType::MNN_STATECACHE_ADVANCED);
    void setHint(int quantType = 0, int type = 1);
    void setFPPrecision(BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal) {
        FP_precision = precision;
    }
    void setConfig(struct StateCacheManagerConfig config);
    MNNStateCacheQuantType getQuantType() const {
        return mQuantType;
    }
    MNNStateCacheType getStateCacheType() const {
        return mType;
    }
    int getBlockSize() const {
        return mBlockSize;
    }
    int getSlotNum(void* layer) const {
        return mCurrentReference->getSlotNum(layer);
    }


    // Reference correlated stuff
    std::shared_ptr<StateCacheReference> getCurrentReference() {return mCurrentReference;}
    void setCurrentReference(std::shared_ptr<StateCacheReference> other) {mCurrentReference = other;}
    std::shared_ptr<StateCacheReference> onCreateReference(bool from_current=false);
    void* onCreateIdentifier(void* identifier=nullptr) {
        if (identifier!=nullptr) return identifier;
        else return (void*)(size_t)(mNextNewLayerId++);
    }

    // Enlarge the memory resources
    bool enlargeMemCache(void* layer, size_t size);
    bool enlargeFileCache(void* layer, size_t size);
    // Release the memory resources
    void releaseBlockMem(void* layer, std::shared_ptr<StateCacheBlock> block);
    void releaseMemCache();
    void releaseFileCache();
    // Evict a block
    std::shared_ptr<StateCacheBlock> evictBlock(const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list);
    // Get a free pointer
    // Assume that block sizes are all the same among all layers!
    char* getFreePtr(void* layer, size_t size);
    char* getFreePtr(void* layer, size_t size, const std::vector<std::shared_ptr<StateCacheBlock>>& evict_pin_block_list);
    // Recover a block from the file
    void recoverBlock(void* layer, std::shared_ptr<StateCacheBlock> block);
    void recoverBlock(void* layer, std::shared_ptr<StateCacheBlock> block, const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list);
    // Desert a block
    void desertBlock(void* layer, int ref_id, std::shared_ptr<StateCacheBlock> block);
    // Copy a block
    std::shared_ptr<StateCacheBlock> copyBlock(int ref_id, std::shared_ptr<StateCacheBlock> block_ptr, const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list);

    // external calls
    void onAllocateCache(void* layer, void* backend, int token_num, std::vector<std::vector<int>> shape, int hP);
    int prepareAttn(void* layer, int previous_token_num, std::vector<std::shared_ptr<StateCacheBlock>>& pastKV);
    void postAttn(void* layer, int last_block_slot_num); 

    // destructor
    void clear();
    ~StateCacheManager() {printf("error!\n"); clear();}
};

} // namespace MNN

#endif // StateCacheManager_hpp