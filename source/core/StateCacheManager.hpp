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
#include <MNN/Tensor.hpp>
#include <iostream>
#include <functional> 

namespace PagedAttention {

template<typename T>
class Optional {
public:
    Optional() : m_value(nullptr) {}

    explicit Optional(T value) : m_value(new T(value)) {}

    ~Optional() {
        delete m_value;
    }

    bool hasValue() const {
        return m_value != nullptr;
    }

    T& value() {
        if (!hasValue()) {
            throw std::runtime_error("Optional does not contain a value.");
        }
        return *m_value;
    }

    const T& value() const {
        if (!hasValue()) {
            throw std::runtime_error("Optional does not contain a value.");
        }
        return *m_value;
    }

    T* operator->() {
        if (!hasValue()) {
            throw std::runtime_error("Optional does not contain a value.");
        }
        return m_value;
    }

    const T* operator->() const {
        if (!hasValue()) {
            throw std::runtime_error("Optional does not contain a value.");
        }
        return m_value;
    }

private:
    T* m_value;
};


// 2.1 StateCacheBlock
struct StateCacheBlock {
    std::vector<int> ref_ids; // IDs of samples using this block
    Optional<int> slot_end; // Index pointing to the next empty slot, or empty if full
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
        inMemBlockList {[](const std::shared_ptr<StateCacheBlock>& a, const std::shared_ptr<StateCacheBlock>& b) { return a->ref_ids.size() > b->ref_ids.size(); }};

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

} // namespace PagedAttention

#endif // StateCacheManager_hpp