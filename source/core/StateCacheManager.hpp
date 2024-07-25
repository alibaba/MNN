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
#include <optional>
#include <cassert>
#include <MNN/Tensor.hpp>

namespace PagedAttention {

// 2.1 StateCacheBlock
struct StateCacheBlock {
    std::vector<int> ref_ids; // IDs of samples using this block
    std::optional<int> slot_end; // Index pointing to the next empty slot, or empty if full
    std::shared_ptr<MNN::Tensor*> slots; // Tensor holding the KV cache data
};

// 2.2 StateCache
struct StateCache {
    // List of pointers to free memory blocks
    std::list<std::shared_ptr<StateCacheBlock>> freePtrList;

    // List of offsets in external storage for free blocks
    std::list<size_t> freeFileOffsetList;

    // Dynamic structure for in-memory blocks with minimal ref_ids size for eviction
    std::priority_queue<std::shared_ptr<StateCacheBlock>, std::vector<std::shared_ptr<StateCacheBlock>>, 
                        std::function<bool(const std::shared_ptr<StateCacheBlock>&, const std::shared_ptr<StateCacheBlock>&)>> 
        inMemBlockList {[](const auto& a, const auto& b) { return a->ref_ids.size() > b->ref_ids.size(); }};

    // Linked list of blocks currently being used for computation
    std::list<std::shared_ptr<StateCacheBlock>> computeCacheBlockList;

    // Linked list of blocks stored in external storage
    std::list<std::shared_ptr<StateCacheBlock>> offloadedCacheBlockList;
};

// 2.3 StateCacheManager
class StateCacheManager {
public:
    StateCache state_cache;

    // Enlarge the memory cache
    virtual bool enlargeMemCache(size_t size) = 0;

    // Enlarge the file cache
    virtual bool enlargeFileCache(size_t size) = 0;

    // Release the memory cache
    virtual void releasesMemCache() = 0;

    // Release the file cache
    virtual void releaseFileCache() = 0;

    // Evict a block
    virtual std::shared_ptr<StateCacheBlock> evictBlock(const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list) = 0;

    // Get a free pointer
    virtual std::shared_ptr<StateCacheBlock> getFreePtr(const std::vector<std::shared_ptr<StateCacheBlock>>& evict_pin_block_list) = 0;

    // Recover a block from the file
    virtual void recoverBlock(std::shared_ptr<StateCacheBlock> block_ptr, const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list) = 0;

    // Desert a block
    virtual void desertBlock(int ref_id, std::shared_ptr<StateCacheBlock> block_ptr) = 0;

    // Copy a block
    virtual std::shared_ptr<StateCacheBlock> copyBlock(int ref_id, std::shared_ptr<StateCacheBlock> block_ptr) = 0;

    // Prepare attention
    virtual void prepareAttn(int ref_id, const std::vector<std::shared_ptr<StateCacheBlock>>& argv) = 0;
};

} // namespace PagedAttention

#endif // StateCacheManager_hpp