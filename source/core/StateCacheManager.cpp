#include <cstdio>
#include <fcntl.h> // For O_RDWR, O_CREAT, etc.
#include <sys/stat.h> // For S_IRUSR, S_IWUSR
#include <sys/types.h> // For mode_t
#include <unistd.h> // For open, ftruncate, close, pread, pwrite
#include <cstring>
#include <algorithm>

#include "core/StateCacheManager.hpp"

namespace MNN {

/* -------------StateCacheBlock------------ */
StateCacheBlock::StateCacheBlock(std::vector<int> ref_ids, int blok_size, int slot_num) : mRefIds(ref_ids), mBlockSize(blok_size), mSlotNum(slot_num) {
    mTensors.resize(mBlockSize);    
    mTensorSize.resize(mBlockSize);
}
// add and get Tensor
void StateCacheBlock::setTensor(int tId, Tensor* tensor) {
    mTensors[tId] = tensor;
}
void StateCacheBlock::setTensorSize(int tId, int tensor_size) {
    mTensorSize[tId] = tensor_size;
}
// manage pointers and offsets
bool StateCacheBlock::onAllocatePtr(uint8_t* ptr) {
    for (int i=0; i < mTensors.size(); ++i) {
        // 1. allocate
        mTensors[i]->buffer().host = ptr;
        // 2. add the tensor size to ptr. 
        ptr += mTensorSize[i];
    }
}
bool StateCacheBlock::onAllocateOffset(size_t offset) {
    for (int i=0; i < mTensors.size(); ++i) {
        // 1. allocate
        mTensors[i]->setFileOffset(offset);
        // 2. add the tensor size to offset.
        offset += mTensorSize[i];
    }
}
// reset slot_end
void StateCacheBlock::resetSlotNum(int slot_num) {
    mSlotNum = slot_num;
}
    // deal with sample reference
void StateCacheBlock::removeRef(int ref_id) {
    auto it = std::find(mRefIds.begin(),mRefIds.end(),ref_id);
    if (it != mRefIds.end()) { 
        mRefIds.erase(it); 
    } 
}
void StateCacheBlock::addRef(int ref_id) {
    mRefIds.emplace_back(ref_id);
}


/* -------------StateCacheManager------------ */
bool StateCacheManager::enlargeMemCache(size_t size) {
    // Implementation for enlarging the memory cache
    // This could involve allocating more memory and updating the state_cache accordingly
    // For simplicity, let's just return true indicating success
    return true;
}

bool StateCacheManager::enlargeFileCache(size_t size) {
    // Determine the current file path
    std::string currentCacheFilePath = "state_cache_file.bin";
    std::string newCacheFilePath = "state_cache_file_new.bin";

    // Check if the current file exists
    if (std::remove(currentCacheFilePath.c_str()) == 0 || errno == ENOENT) {
        // The file either doesn't exist or was successfully removed, so we're good to proceed
    } else {
        // Handle error: Could not remove file.
        // You can log the error here, for example:
        std::cout << "Warning: Could not remove file " << currentCacheFilePath << "." << std::endl;
        return false;
    }

    // Create the new file with the specified size
    int newFd = open(newCacheFilePath.c_str(), O_RDWR | O_CREAT | O_EXCL | O_DIRECT, S_IRUSR | S_IWUSR);
    if (newFd == -1) {
        // Handle error: Could not create the new file.
        // You can log the error here, for example:
        std::cout << "Warning: Could not create file " << newCacheFilePath << "." << std::endl;
        return false;
    }

    // Ensure the new file has at least the requested size
    if (ftruncate(newFd, size) == -1) {
        // Handle error: Could not set the file size.
        // You can log the error here, for example:
        std::cout << "Warning: Could not set the size of file " << newCacheFilePath << "." << std::endl;
        close(newFd);
        return false;
    }

    // If the current file exists, copy its contents to the new file
    if (std::remove(currentCacheFilePath.c_str()) == 0 || errno == ENOENT) {
        // The file either doesn't exist or was successfully removed, so we're good to proceed
    } else {
        // The current file exists, so we need to copy its contents to the new file
        int currentFd = open(currentCacheFilePath.c_str(), O_RDONLY | O_DIRECT);
        if (currentFd == -1) {
            // Handle error: Could not open the current file.
            // You can log the error here, for example:
            std::cout << "Warning: Could not open file " << currentCacheFilePath << "." << std::endl;
            close(newFd);
            return false;
        }

        // Read the current file content and write it to the new file
        char buffer[4096];
        ssize_t bytesRead;
        while ((bytesRead = pread(currentFd, buffer, sizeof(buffer), 0)) > 0) {
            ssize_t bytesWritten = pwrite(newFd, buffer, bytesRead, 0);
            if (bytesWritten != bytesRead) {
                // Handle error: Could not write all bytes to the new file.
                // You can log the error here, for example:
                std::cout << "Warning: Could not write all bytes to file " << newCacheFilePath << "." << std::endl;
                close(currentFd);
                close(newFd);
                return false;
            }
        }

        close(currentFd);
    }

    // Rename the new file to overwrite the old file
    if (rename(newCacheFilePath.c_str(), currentCacheFilePath.c_str()) == -1) {
        // Handle error: Could not rename the file.
        // You can log the error here, for example:
        std::cout << "Warning: Could not rename file " << newCacheFilePath << " to " << currentCacheFilePath << "." << std::endl;
        close(newFd);
        return false;
    }

    close(newFd);

    return true;
}

void StateCacheManager::releasesMemCache() {
     // Clear the inMemBlockList, which will destroy the shared pointers and release the memory.
    while (!state_cache.inMemBlockList.empty()) {
        auto block = state_cache.inMemBlockList.top();
        state_cache.inMemBlockList.pop();
        // No need to explicitly delete the block as the shared_ptr will handle it.
    }

    // Clear the computeCacheBlockList, which will destroy the shared pointers and release the memory.
    for (auto& block : state_cache.computeCacheBlockList) {
        // No need to explicitly delete the block as the shared_ptr will handle it.
    }
    state_cache.computeCacheBlockList.clear();

    // Clear the freePtrList, which will destroy the shared pointers and release the memory.
    for (auto& block : state_cache.freePtrList) {
        // No need to explicitly delete the block as the shared_ptr will handle it.
    }
    state_cache.freePtrList.clear();
}

void StateCacheManager::releaseFileCache() {
    // Assuming there is a file path stored somewhere in the StateCache or StateCacheManager
    std::string cacheFilePath = "state_cache_file.bin"; // Example file path

    // Check if the file exists before attempting to remove it
    if (remove(cacheFilePath.c_str()) != 0) {
        // Handle error: Could not remove file.
        // You can log the error here, for example:
        std::cout << "Warning: Could not remove file " << cacheFilePath << "." << std::endl;
    } else {
        // File was successfully removed, now close the file descriptor if it's open
        int fd = open(cacheFilePath.c_str(), O_RDWR | O_DIRECT);
        if (fd != -1) {
            if (close(fd) == -1) {
                // Handle error: Could not close file.
                // You can log the error here, for example:
                std::cout << "Warning: Could not close file " << cacheFilePath << "." << std::endl;
            }
        } else {
            // File descriptor could not be opened, likely because the file was already removed.
        }
    }

    // Clear the list of offsets in external storage for free blocks
    state_cache.freeFileOffsetList.clear();
}

std::shared_ptr<StateCacheBlock> StateCacheManager::evictBlock(const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list) {
    std::shared_ptr<StateCacheBlock> evict_block;

// Find a block to evict from computeCacheBlockList
for (auto& block : state_cache.computeCacheBlockList) {
    bool is_pinned = false;
    for (const auto& pinned_block : pin_block_list) {
        if (block.get() == pinned_block.get()) {
            is_pinned = true;
            break;
        }
    }
    if (!is_pinned) {
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
    // std::shared_ptr<StateCacheBlock> free_ptr = getFreePtr(pin_block_list);

    // if (free_ptr) {
    //     // Assuming block size is known and fixed
    //     constexpr size_t BLOCK_SIZE = 1024; // Example block size

    //     // Ensure the file is open for direct IO
    //     if (state_cache.file_fd < 0) {
    //         state_cache.file_fd = ::open("cache_file", O_RDWR | O_DIRECT, S_IRUSR | S_IWUSR);
    //     }

    //     // Read block_ptr from disk
    //     ::pread(state_cache.file_fd, block_ptr->slots.get(), BLOCK_SIZE, block_ptr->slot_end->value());

    //     // Write block_ptr to free_ptr
    //     ::memcpy(free_ptr->slots.get(), block_ptr->slots.get(), BLOCK_SIZE);

    //     // Add the block's offset back to freeFileOffsetList
    //     state_cache.freeFileOffsetList.push_back(block_ptr->slot_end->value());
    // }
}

void StateCacheManager::desertBlock(int ref_id, std::shared_ptr<StateCacheBlock> block_ptr) {
    // Find the ref_id in the vector
    block_ptr->removeRef(ref_id);

    if (block_ptr->needDesert()) {
        // Block is no longer referenced, free it
        state_cache.freePtrList.push_back(block_ptr);
    } else {
        // Update the inMemBlockList
        state_cache.inMemBlockList.push(block_ptr);
    }
}

std::shared_ptr<StateCacheBlock> StateCacheManager::copyBlock(int ref_id, std::shared_ptr<StateCacheBlock> block_ptr) {
//     // TODO: This is incorrect!!!!!
//     auto new_block = std::make_shared<StateCacheBlock>(*block_ptr);
//     new_block->ref_ids = {ref_id}; // Change the ref_ids
//     return new_block;
}

void StateCacheManager::prepareAttn(int ref_id, const std::vector<std::shared_ptr<StateCacheBlock>>& argv) {
    // Logic to bring input/output blocks into memory
    // This could involve calling getFreePtr, evictBlock, and recoverBlock
    // For simplicity, let's just outline the steps
    // for (const auto& arg : argv) {
    //     if (arg->ref_ids.size() > 1) {
    //         // Copy the block if it has more than one ref_id
    //         std::shared_ptr<StateCacheBlock> new_block = copyBlock(ref_id, arg);
    //         // Bring the new block into memory
    //         recoverBlock(new_block, argv);
    //     } else {
    //         // Bring the block into memory
    //         recoverBlock(arg, argv);
    //     }
    // }
}

} // namespace MNN