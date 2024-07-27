#include "core/StateCacheManager.hpp"

namespace MNN {

StateCacheBlock::StateCacheBlock(std::vector<int> ref_ids, int blok_size, int slot_num) : mRefIds(ref_ids), mBlockSize(blok_size), mSlotNum(slot_num) {
    mBackend.reset(ExecutorScope::Current()->getAttr()->constantBackend);
    mTensors.resize(mBlockSize);    
}
// add and get Tensor
void StateCacheBlock::setTensor(int tId, Tensor* tensor) {
    mTensors[tId] = tensor;
}
// manage pointers and offsets
bool onAllocatePtr(uint8_t* ptr) {
    for (auto tensor : mTensors) {
        // 1. allocate
        tensor->buffer().host = ptr;
        // 2. add the tensor size to ptr. 
        pointer += mBackend->getTensorSize(tensor, true);
    }
}
bool onAllocateOffset(size_t offset) {
    for (auto tensor : mTensors) {
        // 1. allocate
        tensor->setFileOffset(offset);
        // 2. add the tensor size to offset.
        offset += mBackend->getTensorSize(tensor, true);
    }
}
// reset slot_end
void resetSlotNum(int slot_num) {
    mSlotNum = slot_num;
}

bool StateCacheManager::enlargeMemCache(size_t size) {
    // Implementation for enlarging the memory cache
    // This could involve allocating more memory and updating the state_cache accordingly
    // For simplicity, let's just return true indicating success
    return true;
}

bool StateCacheManager::enlargeFileCache(size_t size) {
    // Implementation for enlarging the file cache
    // This could involve creating a new file and copying existing data
    // For simplicity, let's just return true indicating success
    return true;
}

void StateCacheManager::releasesMemCache() {
    // Implementation for releasing the memory cache
    // This could involve freeing the allocated memory
}

void StateCacheManager::releaseFileCache() {
    // Implementation for releasing the file cache
    // This could involve closing and removing the file
}

std::shared_ptr<StateCacheBlock> StateCacheManager::evictBlock(const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list) {
    std::shared_ptr<StateCacheBlock> evict_block;

    // Find a block to evict from computeCacheBlockList
    for (auto& block : state_cache.computeCacheBlockList) {
        if (std::find(pin_block_list.begin(), pin_block_list.end(), block) == pin_block_list.end()) {
            evict_block = block;
            break;
        }
    }

    // If no block was found, find a block from inMemBlockList
    if (!evict_block) {
        if (!state_cache.inMemBlockList.empty()) {
            evict_block = state_cache.inMemBlockList.top();
            state_cache.inMemBlockList.pop();
        }
    }

    if (evict_block) {
        // Get a file offset from freeFileOffsetList
        if (state_cache.freeFileOffsetList.empty()) {
            if (!enlargeFileCache(0)) {
                // Enlargement failed
                return nullptr;
            }
        }
        size_t offset = *state_cache.freeFileOffsetList.begin();
        state_cache.freeFileOffsetList.pop_front();

        // Move the evicted block to offloadedCacheBlockList
        state_cache.offloadedCacheBlockList.push_back(evict_block);
        state_cache.freePtrList.push_back(evict_block);

        // Remove the block from computeCacheBlockList or inMemBlockList
        state_cache.computeCacheBlockList.remove(evict_block);
        return evict_block;
    }

    return nullptr;
}

std::shared_ptr<StateCacheBlock> StateCacheManager::getFreePtr(const std::vector<std::shared_ptr<StateCacheBlock>>& evict_pin_block_list) {
    if (state_cache.freePtrList.empty()) {
        if (!enlargeMemCache(0)) {
            // Enlargement failed
            return nullptr;
        }
    }

    std::shared_ptr<StateCacheBlock> free_ptr = *state_cache.freePtrList.begin();
    state_cache.freePtrList.pop_front();

    return free_ptr;
}

void StateCacheManager::recoverBlock(std::shared_ptr<StateCacheBlock> block_ptr, const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list) {
    std::shared_ptr<StateCacheBlock> free_ptr = getFreePtr(pin_block_list);

    if (free_ptr) {
        // Write block_ptr to free_ptr
        // This could involve reading from the file and writing to the memory
        // For simplicity, let's just assume it works

        // Add the block's offset back to freeFileOffsetList
        state_cache.freeFileOffsetList.push_back(block_ptr->slot_end.value_or(0));
    }
}

void StateCacheManager::desertBlock(int ref_id, std::shared_ptr<StateCacheBlock> block_ptr) {
    auto it = std::find(block_ptr->ref_ids.begin(), block_ptr->ref_ids.end(), ref_id);
    if (it != block_ptr->ref_ids.end()) {
        block_ptr->ref_ids.erase(it);

        if (block_ptr->ref_ids.empty()) {
            // Block is no longer referenced, free it
            state_cache.freePtrList.push_back(block_ptr);
        } else {
            // Update the inMemBlockList
            state_cache.inMemBlockList.push(block_ptr);
        }
    }
}

std::shared_ptr<StateCacheBlock> StateCacheManager::copyBlock(int ref_id, std::shared_ptr<StateCacheBlock> block_ptr) {
    auto new_block = std::make_shared<StateCacheBlock>(*block_ptr);
    new_block->ref_ids = {ref_id}; // Change the ref_ids
    return new_block;
}

void StateCacheManager::prepareAttn(int ref_id, const std::vector<std::shared_ptr<StateCacheBlock>>& argv) {
    // Logic to bring input/output blocks into memory
    // This could involve calling getFreePtr, evictBlock, and recoverBlock
    // For simplicity, let's just outline the steps
    for (const auto& arg : argv) {
        if (arg->ref_ids.size() > 1) {
            // Copy the block if it has more than one ref_id
            std::shared_ptr<StateCacheBlock> new_block = copyBlock(ref_id, arg);
            // Bring the new block into memory
            recoverBlock(new_block, argv);
        } else {
            // Bring the block into memory
            recoverBlock(arg, argv);
        }
    }
}

} // namespace MNN