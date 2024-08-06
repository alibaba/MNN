#include <cstdio>
#include <fcntl.h> // For O_RDWR, O_CREAT, etc.
#include <sys/stat.h> // For S_IRUSR, S_IWUSR
#include <sys/types.h> // For mode_t
#include <unistd.h> // For open, ftruncate, close, pread, pwrite
#include <cstring>
#include <algorithm>
#include <filesystem>

#include <MNN/Tensor.hpp>
#include <MNN/StateCacheManager.hpp>
#include "Macro.h"
#include "Backend.hpp"


namespace MNN {

/* -------------StateCacheBlock------------ */
StateCacheBlock::StateCacheBlock(std::set<int> ref_ids, int blok_size, int slot_num) : mRefIds(ref_ids), mBlockSize(blok_size), mSlotNum(slot_num) {
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
void StateCacheBlock::resetTensorShape(std::vector<std::vector<int>>& shape, int hp) {
    for (auto& s: shape){
        s[s.size()-2] = UP_DIV(mBlockSize, hp);
        s.push_back(hp);
    }
}
size_t StateCacheBlock::setTensors(std::vector<std::vector<int>>& shape, void* backend, MNNStateCacheQuantType type, int hp) {
    resetTensorShape(shape, hp);
    if (type == MNNStateCacheQuantType::NoQuant) {
        setTensor(LAYOUT::NoQuant::PAST_K, Tensor::createDevice<float>(shape[LAYOUT::NoQuant::PAST_K]));
        setTensor(LAYOUT::NoQuant::PAST_V, Tensor::createDevice<float>(shape[LAYOUT::NoQuant::PAST_V]));
        return;
    }
    if (type == MNNStateCacheQuantType::QuantKeyInt8) {
        setTensor(LAYOUT::QuantKeyInt8::PAST_K, Tensor::createDevice<int8_t>(shape[LAYOUT::QuantKeyInt8::PAST_K]));
        setTensor(LAYOUT::QuantKeyInt8::PAST_K_SCALES, Tensor::createDevice<float>(shape[LAYOUT::QuantKeyInt8::PAST_K_SCALES]));
        setTensor(LAYOUT::QuantKeyInt8::PAST_K_ZERO_POINTS, Tensor::createDevice<float>(shape[LAYOUT::QuantKeyInt8::PAST_K_ZERO_POINTS]));
        setTensor(LAYOUT::QuantKeyInt8::PAST_V, Tensor::createDevice<float>(shape[LAYOUT::QuantKeyInt8::PAST_V]));
        return;
    }
    if (type == MNNStateCacheQuantType::QuantValueFp8) {
        setTensor(LAYOUT::QuantValueFp8::PAST_K, Tensor::createDevice<float>(shape[LAYOUT::QuantValueFp8::PAST_K]));
        setTensor(LAYOUT::QuantValueFp8::PAST_V, Tensor::createDevice<uint8_t>(shape[LAYOUT::QuantValueFp8::PAST_V]));
        return;
    }
    if (type == MNNStateCacheQuantType::QuantKeyInt8ValueFp8) {
        setTensor(LAYOUT::QuantKeyInt8ValueFp8::PAST_K, Tensor::createDevice<int8_t>(shape[LAYOUT::QuantKeyInt8ValueFp8::PAST_K]));
        setTensor(LAYOUT::QuantKeyInt8ValueFp8::PAST_K_SCALES, Tensor::createDevice<float>(shape[LAYOUT::QuantKeyInt8ValueFp8::PAST_K_SCALES]));
        setTensor(LAYOUT::QuantKeyInt8ValueFp8::PAST_K_ZERO_POINTS, Tensor::createDevice<float>(shape[LAYOUT::QuantKeyInt8ValueFp8::PAST_K_ZERO_POINTS]));
        setTensor(LAYOUT::QuantKeyInt8ValueFp8::PAST_V, Tensor::createDevice<uint8_t>(shape[LAYOUT::QuantKeyInt8ValueFp8::PAST_V]));
        return;
    }
    if (type == MNNStateCacheQuantType::QuantValueInt8) {
        setTensor(LAYOUT::QuantValueInt8::PAST_K, Tensor::createDevice<float>(shape[LAYOUT::QuantValueInt8::PAST_K]));
        setTensor(LAYOUT::QuantValueInt8::PAST_V, Tensor::createDevice<int8_t>(shape[LAYOUT::QuantValueInt8::PAST_V]));
        setTensor(LAYOUT::QuantValueInt8::PAST_V_SCALES, Tensor::createDevice<float>(shape[LAYOUT::QuantValueInt8::PAST_V_SCALES]));
        setTensor(LAYOUT::QuantValueInt8::PAST_V_ZERO_POINTS, Tensor::createDevice<float>(shape[LAYOUT::QuantValueInt8::PAST_V_ZERO_POINTS]));
        return;
    }
    if (type == MNNStateCacheQuantType::QuantKeyInt8ValueInt8) {
        setTensor(LAYOUT::QuantKeyInt8ValueInt8::PAST_K, Tensor::createDevice<int8_t>(shape[LAYOUT::QuantKeyInt8ValueInt8::PAST_K]));
        setTensor(LAYOUT::QuantKeyInt8ValueInt8::PAST_K_SCALES, Tensor::createDevice<float>(shape[LAYOUT::QuantKeyInt8ValueInt8::PAST_K_SCALES]));
        setTensor(LAYOUT::QuantKeyInt8ValueInt8::PAST_K_ZERO_POINTS, Tensor::createDevice<float>(shape[LAYOUT::QuantKeyInt8ValueInt8::PAST_K_ZERO_POINTS]));
        setTensor(LAYOUT::QuantKeyInt8ValueInt8::PAST_V, Tensor::createDevice<int8_t>(shape[LAYOUT::QuantKeyInt8ValueInt8::PAST_V]));
        setTensor(LAYOUT::QuantKeyInt8ValueInt8::PAST_V_SCALES, Tensor::createDevice<float>(shape[LAYOUT::QuantKeyInt8ValueInt8::PAST_V_SCALES]));
        setTensor(LAYOUT::QuantKeyInt8ValueInt8::PAST_V_ZERO_POINTS, Tensor::createDevice<float>(shape[LAYOUT::QuantKeyInt8ValueInt8::PAST_V_ZERO_POINTS]));
        return;
    }
    size_t block_size = 0;
    for (int s = 0; s < mTensors.size(); ++s) {
        block->setTensorSize(s, ((Backend*)backend)->getTensorSize(mTensors[s], true));
        block_size += mTensorSize[s];
    }
    return block_size;
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
void StateCacheBlock::setBlockMem(void* ptr, size_t size) {
    mBasePtr = ptr;
    mSize = size;
}
// reset slot_end
void StateCacheBlock::resetSlotNum(int slot_num) {
    mSlotNum = slot_num;
}
    // deal with sample reference
void StateCacheBlock::removeRef(int ref_id) {
    mRefIds.erase(ref_id); 
}
void StateCacheBlock::addRef(int ref_id) {
    mRefIds.insert(ref_id);
}
void StateCacheBlock::clearRef() {
    mRefIds.clear();
}

/* -------------StateCache------------------- */
void StateCache::clear(std::shared_ptr<StateCacheBlock> block) {
    // remove the block from internal data structures!
    for (auto it = inMemBlockList.begin(); it != inMemBlockList.end(); ++it) {
        it.second.erase(block);
    }
    computeCacheBlockList.erase(block);
    offloadedBlockList.erase(block);
}


/* -------------StateCacheManager------------ */
StateCacheManager::StateCacheManager(MNNStateCacheQuantType quantType, MNNStateCacheType type){
    mNextNewRefId = 0;
    mQuantType = quantType;
    mType = type;
    mStateCache.reset(new StateCache);
}

void StateCacheManager::setHint(MNNStateCacheQuantType quantType, MNNStateCacheType type){
    mQuantType = quantType;
    mType = type;
}

void StateCacheManager::setHint(int quantType, int type){
    mQuantType = (MNNStateCacheQuantType)quantType;
    mType = (MNNStateCacheType)type;
}

std::shared_ptr<StateCacheReference> StateCacheManager::onCreateReference(bool from_current) {
    if (from_current) {
        return std::shared_ptr<StateCacheReference>(new StateCacheReference(mNextNewRefId++, mCurrentReference));
    }
    else {
        return std::shared_ptr<StateCacheReference>(new StateCacheReference(mNextNewRefId++, mBlockSize));
    }
}

void StateCacheManager::setConfig(struct StateCacheManagerConfig config) {
    mConfig = config;
    if (mType == MNNStateCacheType::MNN_STATECACHE_ADVANCED) {
        mBlockSize = mConfig.blockSize;
    }
}


bool StateCacheManager::enlargeMemCache(void* layer, size_t size) {
    std::shared_ptr<StateCacheBlock> cache = mStateCache[layer];
    void* ptr = malloc(size * mConfig.preallocateBlockNum); // try pre-allocation!
    if (ptr == nullptr) {
        ptr = malloc(size); // try allocation!
        if (ptr == nullptr) return false;
        cache->freePtrList.push_back(ptr);
    } else {
        for (int i = 0; i < mCoinfig.preallocateBlockNum; ++i) {
            cache->freePtrList.push_back(ptr);
            ptr += size;
        }
    }
    return true;
}

bool StateCacheManager::enlargeFileCache(size_t size) {
//     // Determine the current file path
//     std::string currentCacheFilePath = "mStateCache_file.bin";
//     std::string newCacheFilePath = "mStateCache_file_new.bin";

//     // Check if the current file exists
//     if (std::remove(currentCacheFilePath.c_str()) == 0 || errno == ENOENT) {
//         // The file either doesn't exist or was successfully removed, so we're good to proceed
//     } else {
//         // Handle error: Could not remove file.
//         // You can log the error here, for example:
//         std::cout << "Warning: Could not remove file " << currentCacheFilePath << "." << std::endl;
//         return false;
//     }

//     // Create the new file with the specified size
//     int newFd = open(newCacheFilePath.c_str(), O_RDWR | O_CREAT | O_EXCL | O_DIRECT, S_IRUSR | S_IWUSR);
//     if (newFd == -1) {
//         // Handle error: Could not create the new file.
//         // You can log the error here, for example:
//         std::cout << "Warning: Could not create file " << newCacheFilePath << "." << std::endl;
//         return false;
//     }

//     //use resize_file
//     std::error_code ec;
//     std::filesystem::resize_file(newCacheFilePath, size, ec);
//     if (ec) {
//         // Handle error: Could not resize the file.
//         std::cout << "Warning: Could not resize file " << newCacheFilePath << ": " << ec.message() << std::endl;
//         close(newFd);
//         return false;
// }

//     // If the current file exists, copy its contents to the new file
//     if (std::remove(currentCacheFilePath.c_str()) == 0 || errno == ENOENT) {
//         // The file either doesn't exist or was successfully removed, so we're good to proceed
//     } else {
//         // The current file exists, so we need to copy its contents to the new file
//         int currentFd = open(currentCacheFilePath.c_str(), O_RDONLY | O_DIRECT);
//         if (currentFd == -1) {
//             // Handle error: Could not open the current file.
//             // You can log the error here, for example:
//             std::cout << "Warning: Could not open file " << currentCacheFilePath << "." << std::endl;
//             close(newFd);
//             return false;
//         }

//         // Read the current file content and write it to the new file
//         char buffer[4096];
//         ssize_t bytesRead;
//         while ((bytesRead = pread(currentFd, buffer, sizeof(buffer), 0)) > 0) {
//             ssize_t bytesWritten = pwrite(newFd, buffer, bytesRead, 0);
//             if (bytesWritten != bytesRead) {
//                 // Handle error: Could not write all bytes to the new file.
//                 // You can log the error here, for example:
//                 std::cout << "Warning: Could not write all bytes to file " << newCacheFilePath << "." << std::endl;
//                 close(currentFd);
//                 close(newFd);
//                 return false;
//             }
//         }

//         close(currentFd);
//     }

//     // Rename the new file to overwrite the old file
//     if (rename(newCacheFilePath.c_str(), currentCacheFilePath.c_str()) == -1) {
//         // Handle error: Could not rename the file.
//         // You can log the error here, for example:
//         std::cout << "Warning: Could not rename file " << newCacheFilePath << " to " << currentCacheFilePath << "." << std::endl;
//         close(newFd);
//         return false;
//     }

//     close(newFd);

//     return true;
}

void StateCacheManager::releaseBlock(void* layer, std::shared_ptr<StateCacheBlock> block) {
    // 1. clear the block from layer.
    layer->clear(block);
    // 2. release the block's memory
    if (block->getBlockPtr()!=nullptr) free(block->getBlockPtr());
}

void StateCacheManager::releaseMemCache() {
    // iterate through all layers
    for (auto it = mStateCache.begin(); it != mStateCache.end(); ++it) {
        auto layer = it.first;
        auto cache = it.second;
        // clear free ptr
        while (!cache->freePtrList.empty()) {
            free(cache->freePtrList.back());
            cache->freePtrList.pop_back();
        }
        // clear inMemBlockList
        for (auto it = cache->inMemBlockList.begin(); it != cache->inMemBlockList.end(); ++it) {
            for (auto block_it = it.second.begin(); block_it != it.second.end(); ++block_it) {
                if ((*block_it)->getBlockPtr() != nullptr) free((*block_it)->getBlockPtr());
            }
        }
        cache->inMemBlockList.clear();
        // clear computeCacheBlockList
        while (!cache->computeCacheBlockList.empty()) {
            if (cache->computeCacheBlockList.back()->getBlockPtr() != nullptr) free(cache->computeCacheBlockList.back()->getBlockPtr());
            cache->computeCacheBlockList.pop_back();
        }
        cache->computeCacheBlockList.clear()
    }
}

void StateCacheManager::releaseFileCache() {
    // // Assuming there is a file path stored somewhere in the StateCache or StateCacheManager
    // std::string cacheFilePath = "mStateCache_file.bin"; // Example file path

    // // Check if the file exists before attempting to remove it
    // if (remove(cacheFilePath.c_str()) != 0) {
    //     // Handle error: Could not remove file.
    //     // You can log the error here, for example:
    //     std::cout << "Warning: Could not remove file " << cacheFilePath << "." << std::endl;
    // } else {
    //     // File was successfully removed, now close the file descriptor if it's open
    //     int fd = open(cacheFilePath.c_str(), O_RDWR | O_DIRECT);
    //     if (fd != -1) {
    //         if (close(fd) == -1) {
    //             // Handle error: Could not close file.
    //             // You can log the error here, for example:
    //             std::cout << "Warning: Could not close file " << cacheFilePath << "." << std::endl;
    //         }
    //     } else {
    //         // File descriptor could not be opened, likely because the file was already removed.
    //     }
    // }

    // // Clear the list of offsets in external storage for free blocks
    // mStateCache->freeFileOffsetList.clear();
}

std::shared_ptr<StateCacheBlock> StateCacheManager::evictBlock(const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list) {

    // std::cout<<"enter-2";
    // std::shared_ptr<StateCacheBlock> evict_block;

    // // Find a block to evict from computeCacheBlockList
    // for (auto& block : state_cache.computeCacheBlockList) {
    //     std::cout<<"enter-1:"<<block<<std::endl;
    //     bool is_pinned = false;
    //     for (const auto& pinned_block : pin_block_list) {
    //         std::cout<<"enter0:"<<pinned_block<<std::endl;
    //         if (block.get() == pinned_block.get()) {
    //             is_pinned = true;
    //             break;
    //         }
    //     }
    //     if (!is_pinned) {
    //         std::cout<<"enter1";
    //         evict_block = block;
    //         break;
    //     }
    // }

    // // If no block was found, find a block from inMemBlockList
    // if (!evict_block) {
    //     std::cout<<"enter2";
    //     if (!state_cache.inMemBlockList.empty()) {
    //         std::cout<<"enter3";
    //         evict_block = state_cache.inMemBlockList.top();
    //         state_cache.inMemBlockList.pop();
    //     }
    // }

    // if (evict_block) {
    //     std::cout<<"enter4";
    //     // Get a file offset from freeFileOffsetList
    //     if (state_cache.freeFileOffsetList.empty()) {
    //         std::cout<<"enter5";
    //         if (!enlargeFileCache(0)) {
    //             // Enlargement failed
    //             std::cout<<"nullptr1";
    //             return nullptr;
    //         }
    //     }
    //     size_t offset = *state_cache.freeFileOffsetList.begin();
    //     state_cache.freeFileOffsetList.pop_front();


    //     // Open the file for writing
    //     std::ofstream file("external_storage.bin", std::ios::binary | std::ios::out | std::ios::app);

    //     if (!file.is_open()) {
    //         std::cout<<"Failed to open the external storage file.";
    //         return nullptr;
    //     }

    //     // Write the tensor data from the evicted block to the file
    //     size_t current_offset = offset;
    //     for (int i = 0; i < evict_block->getTensorsLength(); ++i) {
    //         // Get the size of the current tensor
    //         int tensor_size = evict_block->getTensorSize(i);

    //         // Set the file position to the correct offset
    //         file.seekp(current_offset, std::ios::beg);

    //         // Write the tensor data to the file
    //         file.write(reinterpret_cast<const char*>(evict_block->getTensor(i)->buffer().host), tensor_size);

    //         // Update the file offset for the current tensor
    //         evict_block->getTensor(i)->setFileOffset(current_offset);

    //         // Update the current offset for the next tensor
    //         current_offset += tensor_size;

    //         // Clear the tensor data in memory
    //         evict_block->getTensor(i)->buffer().host = nullptr; // Mark as not allocated
    //     }

    //     // Close the file
    //     file.close();


    //     // Move the evicted block to offloadedCacheBlockList
    //     state_cache.offloadedCacheBlockList.push_back(evict_block);
    //     state_cache.freePtrList.push_back(evict_block);

    //     // Remove the block from computeCacheBlockList or inMemBlockList
    //     state_cache.computeCacheBlockList.remove(evict_block);
    //     return evict_block;
    // }
    // std::cout<<"nullptr2";
    // return nullptr;
}

void* StateCacheManager::getFreePtr(void* layer, size_t size) {
    std::shared_ptr<StateCacheBlock> cache = mStateCache[layer];
    void* free_ptr = nullptr;
    if (mStateCache->freePtrList.empty()) {
        // No more free pointers. enlarge the MemCache, enlarged ptr will be set!
        if (!enlargeMemCache(layer, size)) {
            // Enlargement failed
            // TODO: add evictions here!! 1. check other layer if they have freePtr (impossible), evict blocks from other layers!
            return free_ptr;
        }
    }
    free_ptr = cache->freePtrList.front();
    cache->freePtrList.pop_front();
    return free_ptr;
}


void* StateCacheManager::getFreePtr(void* layer, size_t size, const std::vector<std::shared_ptr<StateCacheBlock>>& evict_pin_block_list) {
    std::shared_ptr<StateCacheBlock> cache = mStateCache[layer];
    void* free_ptr = nullptr;
    if (mStateCache->freePtrList.empty()) {
        // No more free pointers. enlarge the MemCache, enlarged ptr will be set!
        if (!enlargeMemCache(layer, size)) {
            // Enlargement failed
            // TODO: add evictions here!! 1. check other layer if they have freePtr (impossible), evict blocks from other layers!
            return free_ptr;
        }
    }
    free_ptr = cache->freePtrList.front();
    cache->freePtrList.pop_front();
    return free_ptr;
}

void StateCacheManager::recoverBlock(void* layer, std::shared_ptr<StateCacheBlock> block) {
    if (block->getBlockPtr != nullptr) return;
    // if not in memory, need eviction, notimplemented yet!
}

void StateCacheManager::recoverBlock(std::shared_ptr<StateCacheBlock> block_ptr, const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list) {
    if (block->getBlockPtr != nullptr) return;
    // if not in memory, need eviction, notimplemented yet!

    // // Step 1: Get a free pointer (memory block) using the getFreePtr function
    // std::shared_ptr<StateCacheBlock> free_ptr = getFreePtr(pin_block_list);

    // // Step 2: Write block_ptr from external storage into the free pointer
    // // Open the file for reading
    // std::ifstream file("external_storage.bin", std::ios::binary);

    // if (!file.is_open()) {
    //     throw std::runtime_error("Failed to open the external storage file.");
    // }

    // for (int i = 0; i < block_ptr->getTensorsLength(); ++i) {
    //     // Get the file offset for the current tensor
    //     size_t file_offset = block_ptr->getTensor(i)->getFileOffset();

    //     // Get the size of the current tensor
    //     int tensor_size = block_ptr->getTensorSize(i);

    //     // Set the file position to the correct offset
    //     file.seekg(file_offset, std::ios::beg);

    //     // Read the tensor data from the file
    //     uint8_t* tensor_data = new uint8_t[tensor_size];
    //     file.read(reinterpret_cast<char*>(tensor_data), tensor_size);

    //     // Set the tensor data in the free pointer
    //     free_ptr->setTensor(i, block_ptr->getTensor(i)); // Reuse the same Tensor object
    //     free_ptr->getTensor(i)->buffer().host = tensor_data; // Set the host pointer to the read data

    //     // Update the tensor size in the free pointer
    //     free_ptr->setTensorSize(i, tensor_size);

    //     // Allocate the tensor in the free pointer
    //     free_ptr->onAllocatePtr(free_ptr->getTensor(i)->buffer().host);
    // }

    // // Close the file
    // file.close();

    // // Step 3: Update the block's other information and add it to the in-memory list
    // free_ptr->setRefIds(block_ptr->getRefIds()); 
    // free_ptr->setBlockSize(block_ptr->getBlockSize());
    // free_ptr->resetSlotNum(block_ptr->getSlotNum());
    // state_cache.inMemBlockList.push(free_ptr); // Add the block to the in-memory list

    // // Step 4: Remove the block from the external storage and update the file offset list
    // state_cache.offloadedCacheBlockList.remove(block_ptr); // Remove the block from the external storage list
    // state_cache.freeFileOffsetList.push_back(block_ptr->getTensor(0)->getFileOffset()); // Add the file offset to the free list
}

void StateCacheManager::desertBlock(void* layer, int ref_id, std::shared_ptr<StateCacheBlock> block_ptr) {
    // Find the ref_id in the vector
    // block_ptr->removeRef(ref_id);

    // if (block_ptr->needDesert()) {
    //     // Block is no longer referenced, free it
    //     mStateCache->freePtrList.push_back(block_ptr);
    // } else {
    //     // Update the inMemBlockList
    //     mStateCache->inMemBlockList.push(block_ptr);
    // }
}

std::shared_ptr<StateCacheBlock> StateCacheManager::copyBlock(int ref_id, std::shared_ptr<StateCacheBlock> block_ptr, const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list) {
    // // Step 1: Get a free pointer (memory block) using the getFreePtr function
    // std::shared_ptr<StateCacheBlock> free_ptr = getFreePtr(pin_block_list);

    // // Step 2: Write block_ptr to the free pointer
    // for (int i = 0; i < block_ptr->getTensorsLength(); ++i) {
    //     free_ptr->setTensor(i, block_ptr->getTensor(i));
    //     free_ptr->setTensorSize(i, block_ptr->getTensorSize(i));
    // }
    // // Step 3: Update the block's other information and add it to the in-memory list
    // free_ptr->setRefIds(block_ptr->getRefIds()); 
    // free_ptr->setBlockSize(block_ptr->getBlockSize());
    // free_ptr->resetSlotNum(block_ptr->getSlotNum());
    // state_cache.inMemBlockList.push(free_ptr); 
    // return free_ptr;
        
}

// The Operator requires an allocation
void StateCacheManager::onAllocateCache(void* layer, void* backend, int token_num, std::vector<std::vector<int>> shape, int hp) {
    // 1. check if the layer is registered.
    if (mStateCache.count(layer)==0) {
        mStateCache[layer] = std::shared_ptr(new StateCache);
    }
    if (mCurrentReference->mPageTable.count(layer)==0) {
        std::vector<std::shared_ptr<StateCacheBlock>> table;
        mCurrentReference->mPageTable[layer] = table; 
    }
    // 2. now StateCache exists.
    // 2.1 the advanced one:
    // calculate the number of new blocks in need and get the pointer and set the block.
    int need_token, need_block;
    bool copy_flag = false;
    need_token = token_num;
    if (mCurrentReference->mPageTable[layer].size() != 0) {
        need_token -= mCurrentReference->mPageTable[layer].back()->getFreeSlotNum(); 
    }
    if (mType == MNNStateCacheType::MNN_STATECACHE_ADVANCED) {
        need_block = UP_DIV(need_token, mBlockSize);
    } else {
        // 2.2 the naive one:
        // resize the only block and enlarge it!
        if (need_token > 0){
            // reallocation
            need_block = 1;
            need_token += mBlockSize; // enlargement
            mBlockSize = need_token + mConfig.preallocateTokenNum; // preallocation
            copy_flag = true; 
        } else {
            // no new allocation take place
            need_block = 0; 
            need_token = 0;
            copy_flag = false;
        }
    }
    // allocate the pointers to the page table
    for (int i = 0; i < need_block; ++i) {
        StateCacheBlock* block = new StateCacheBlock({mCurrentReference->mRefId}, mBlockSize, 0);
        size_t block_size = block->setTensors(shape, backend, mQuantType, hp);
        void* free_ptr = getFreePtr(layer, block_size);
        block->onAllocatePtr(free_ptr);
        block->setBlockMem(free_ptr, block_size);
        mCurrentReference->mPageTable[layer].push_back(std::shared_ptr(block));
        need_token -= mBlockSize;
    }

    // reallocate phase, memcpy
    if (mType == MNNStateCacheType::MNN_STATECACHE_NAIVE && copy_flag) {
        // previous: mPageTable[layer][0], present: mPageTable[layer][1]
        // after copy: mPageTable[layer].pop_front()
        std::memcpy(mCurrentReference->mPageTable[layer][1]->getBlockPtr(), // dst
                    mCurrentReference->mPageTable[layer][0]->getBlockPtr(), // src
                    mCurrentReference->mPageTable[layer][0]->getBlockPhysicalSize() /* size */);
        releaseBlockMem(layer, mCurrentReference->mPageTable[layer][0]);
        mCurrentReference->mPageTable[layer].pop_front();
    }
}

int StateCacheManager::prepareAttn(void* layer, int previous_token_num, std::vector<std::shared_ptr<StateCacheBlock>>& pastKV) {
    pastKV.clear();
    pastKV.resize(mCurrentReference->mPageTable[layer].size());
    for (int i = 0; i < pastKV.size(); ++i) {
        recoverBlock(layer, mCurrentReference->mPageTable[layer][i]);
        pastKV[i] = mCurrentReference->mPageTable[layer][i];
    }
    return previous_token_num / mBlockSize; // the index of the first available block.
}

void StateCacheManager::postAttn(void* layer, int last_block_slot_num) {
    for (int i = 0; i < mCurrentReference->mPageTable[layer].size()-1; ++i) {
        mCurrentReference->mPageTable[layer][i]->resetSlotNum(mBlockSize);
    }
    mCurrentReference->mPageTable[layer][mCurrentReference->mPageTable[layer].size()-1]->resetSlotNum(last_block_slot_num);
}

} // namespace MNN