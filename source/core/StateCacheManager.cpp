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
void StateCacheBlock::resetTensorShape(std::vector<std::vector<int>>& shape, int hp) {
    for (auto& s: shape){
        s[s.size()-2] = UP_DIV(mBlockSize, hp);
        s.push_back(hp);
    }
}
void StateCacheBlock::setTensors(std::vector<std::vector<int>>& shape, MNNStateCacheQuantType type, int hp) {
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
void StateCacheBlock::clearRef() {
    mRefIds.clear();
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

bool StateCacheManager::enlargeMemCache(size_t size) {
    // Calculate the number of new blocks needed based on the requested size
    int DEFAULT_BLOCK_SIZE = 4;
    int newBlocksNeeded = size / DEFAULT_BLOCK_SIZE;
    if (size % DEFAULT_BLOCK_SIZE != 0) {
        newBlocksNeeded++; // Add one more block if there is a remainder
    }

    // Create new StateCacheBlock instances and add them to the freePtrList
    for (int i = 0; i < newBlocksNeeded; ++i) {
        // Assuming each block has a size of 4 (this should be the block size you want to use)
        std::vector<int> ref_ids; // Initialize an empty vector of reference IDs
        int block_size = DEFAULT_BLOCK_SIZE; // Use the block size you want to allocate
        int slot_num = 0; // Initialize the slot number to 0
        auto newBlock = std::make_shared<StateCacheBlock>(ref_ids, block_size, slot_num);
        state_cache.freePtrList.push_back(newBlock);
    }


    // If everything went well, return true
    return true;
}

bool StateCacheManager::enlargeFileCache(size_t size) {
    // Determine the current file path
    std::string currentCacheFilePath = "mStateCache_file.bin";
    std::string newCacheFilePath = "mStateCache_file_new.bin";

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

    //use resize_file
    std::error_code ec;
    std::filesystem::resize_file(newCacheFilePath, size, ec);
    if (ec) {
        // Handle error: Could not resize the file.
        std::cout << "Warning: Could not resize file " << newCacheFilePath << ": " << ec.message() << std::endl;
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
    while (!mStateCache->inMemBlockList.empty()) {
        auto block = mStateCache->inMemBlockList.top();
        mStateCache->inMemBlockList.pop();
        // No need to explicitly delete the block as the shared_ptr will handle it.
    }

    // Clear the computeCacheBlockList, which will destroy the shared pointers and release the memory.
    for (auto& block : mStateCache->computeCacheBlockList) {
        // No need to explicitly delete the block as the shared_ptr will handle it.
    }
    mStateCache->computeCacheBlockList.clear();

    // Clear the freePtrList, which will destroy the shared pointers and release the memory.
    for (auto& block : mStateCache->freePtrList) {
        // No need to explicitly delete the block as the shared_ptr will handle it.
    }
    mStateCache->freePtrList.clear();
}

void StateCacheManager::releaseFileCache() {
    // Assuming there is a file path stored somewhere in the StateCache or StateCacheManager
    std::string cacheFilePath = "mStateCache_file.bin"; // Example file path

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
    mStateCache->freeFileOffsetList.clear();
}

std::shared_ptr<StateCacheBlock> StateCacheManager::evictBlock(const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list) {

    std::cout<<"enter-2";
    std::shared_ptr<StateCacheBlock> evict_block;

    // Find a block to evict from computeCacheBlockList
    for (auto& block : state_cache.computeCacheBlockList) {
        std::cout<<"enter-1:"<<block<<std::endl;
        bool is_pinned = false;
        for (const auto& pinned_block : pin_block_list) {
            std::cout<<"enter0:"<<pinned_block<<std::endl;
            if (block.get() == pinned_block.get()) {
                is_pinned = true;
                break;
            }
        }
        if (!is_pinned) {
            std::cout<<"enter1";
            evict_block = block;
            break;
        }
    }

    // If no block was found, find a block from inMemBlockList
    if (!evict_block) {
        std::cout<<"enter2";
        if (!state_cache.inMemBlockList.empty()) {
            std::cout<<"enter3";
            evict_block = state_cache.inMemBlockList.top();
            state_cache.inMemBlockList.pop();
        }
    }

    if (evict_block) {
        std::cout<<"enter4";
        // Get a file offset from freeFileOffsetList
        if (state_cache.freeFileOffsetList.empty()) {
            std::cout<<"enter5";
            if (!enlargeFileCache(0)) {
                // Enlargement failed
                std::cout<<"nullptr1";
                return nullptr;
            }
        }
        size_t offset = *state_cache.freeFileOffsetList.begin();
        state_cache.freeFileOffsetList.pop_front();


        // Open the file for writing
        std::ofstream file("external_storage.bin", std::ios::binary | std::ios::out | std::ios::app);

        if (!file.is_open()) {
            std::cout<<"Failed to open the external storage file.";
            return nullptr;
        }

        // Write the tensor data from the evicted block to the file
        size_t current_offset = offset;
        for (int i = 0; i < evict_block->getTensorsLength(); ++i) {
            // Get the size of the current tensor
            int tensor_size = evict_block->getTensorSize(i);

            // Set the file position to the correct offset
            file.seekp(current_offset, std::ios::beg);

            // Write the tensor data to the file
            file.write(reinterpret_cast<const char*>(evict_block->getTensor(i)->buffer().host), tensor_size);

            // Update the file offset for the current tensor
            evict_block->getTensor(i)->setFileOffset(current_offset);

            // Update the current offset for the next tensor
            current_offset += tensor_size;

            // Clear the tensor data in memory
            evict_block->getTensor(i)->buffer().host = nullptr; // Mark as not allocated
        }

        // Close the file
        file.close();


        // Move the evicted block to offloadedCacheBlockList
        state_cache.offloadedCacheBlockList.push_back(evict_block);
        state_cache.freePtrList.push_back(evict_block);

        // Remove the block from computeCacheBlockList or inMemBlockList
        state_cache.computeCacheBlockList.remove(evict_block);
        return evict_block;
    }
    std::cout<<"nullptr2";
    return nullptr;
}


std::shared_ptr<StateCacheBlock> StateCacheManager::getFreePtr(const std::vector<std::shared_ptr<StateCacheBlock>>& evict_pin_block_list) {
    std::cout<<"enter10";
    if (state_cache.freePtrList.empty()) {
        std::cout<<"enter11";
        if (!enlargeMemCache(0)) {
            std::cout<<"enter12";
            // Enlargement failed
            return nullptr;
        }
    }
std::cout<<"enter13";
    std::shared_ptr<StateCacheBlock> free_ptr = *state_cache.freePtrList.begin();
    state_cache.freePtrList.pop_front();
std::cout<<"enter14";
    return free_ptr;
}

void StateCacheManager::recoverBlock(std::shared_ptr<StateCacheBlock> block_ptr, const std::vector<std::shared_ptr<StateCacheBlock>>& pin_block_list) {
    // Step 1: Get a free pointer (memory block) using the getFreePtr function
    std::shared_ptr<StateCacheBlock> free_ptr = getFreePtr(pin_block_list);

    // Step 2: Write block_ptr from external storage into the free pointer
    // Open the file for reading
    std::ifstream file("external_storage.bin", std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open the external storage file.");
    }

    for (int i = 0; i < block_ptr->getTensorsLength(); ++i) {
        // Get the file offset for the current tensor
        size_t file_offset = block_ptr->getTensor(i)->getFileOffset();

        // Get the size of the current tensor
        int tensor_size = block_ptr->getTensorSize(i);

        // Set the file position to the correct offset
        file.seekg(file_offset, std::ios::beg);

        // Read the tensor data from the file
        uint8_t* tensor_data = new uint8_t[tensor_size];
        file.read(reinterpret_cast<char*>(tensor_data), tensor_size);

        // Set the tensor data in the free pointer
        free_ptr->setTensor(i, block_ptr->getTensor(i)); // Reuse the same Tensor object
        free_ptr->getTensor(i)->buffer().host = tensor_data; // Set the host pointer to the read data

        // Update the tensor size in the free pointer
        free_ptr->setTensorSize(i, tensor_size);

        // Allocate the tensor in the free pointer
        free_ptr->onAllocatePtr(free_ptr->getTensor(i)->buffer().host);
    }

    // Close the file
    file.close();

    // Step 3: Update the block's other information and add it to the in-memory list
    free_ptr->setRefIds(block_ptr->getRefIds()); 
    free_ptr->setBlockSize(block_ptr->getBlockSize());
    free_ptr->resetSlotNum(block_ptr->getSlotNum());
    state_cache.inMemBlockList.push(free_ptr); // Add the block to the in-memory list

    // Step 4: Remove the block from the external storage and update the file offset list
    state_cache.offloadedCacheBlockList.remove(block_ptr); // Remove the block from the external storage list
    state_cache.freeFileOffsetList.push_back(block_ptr->getTensor(0)->getFileOffset()); // Add the file offset to the free list
}

void StateCacheManager::desertBlock(int ref_id, std::shared_ptr<StateCacheBlock> block_ptr) {
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
    // Step 1: Get a free pointer (memory block) using the getFreePtr function
    std::shared_ptr<StateCacheBlock> free_ptr = getFreePtr(pin_block_list);

    // Step 2: Write block_ptr to the free pointer
    for (int i = 0; i < block_ptr->getTensorsLength(); ++i) {
        free_ptr->setTensor(i, block_ptr->getTensor(i));
        free_ptr->setTensorSize(i, block_ptr->getTensorSize(i));
    }
    // Step 3: Update the block's other information and add it to the in-memory list
    free_ptr->setRefIds(block_ptr->getRefIds()); 
    free_ptr->setBlockSize(block_ptr->getBlockSize());
    free_ptr->resetSlotNum(block_ptr->getSlotNum());
    state_cache.inMemBlockList.push(free_ptr); 
    return free_ptr;
        
}

// The Operator requires an allocation
void StateCacheManager::onAllocateCache(void* layer, int token_num, std::vector<int> size, std::vector<std::vector<int>> shape) {
    // 1. check if the layer is registered.
    if (mStateCache.count(layer)==0) {
        mStateCache[layer] = std::shared_ptr(new StateCache);
    }
    std::shared_ptr<StateCache> cache = mStateCache[layer];
    if (mCurrentReference->mPageTable.count(layer)==0) {
        std::vector<std::shared_ptr<StateCacheBlock>> table;
        mCurrentReference->mPageTable[layer] = table; 
    }
    // 2. now StateCache exists.
    // 2.1 the advanced one:
    // calculate the number of new blocks in need and get the pointer and set the block.
    if (mType == MNNStateCacheType::MNN_STATECACHE_ADVANCED) {
        int need_token = token_num;
        if (mCurrentReference->mPageTable[layer].size() != 0) {
            need_token -= (mCurrentReference->mPageTable[layer].back()->getBlockSize() - mCurrentReference->mPageTable[layer].back()->getSlotNum()); 
        }
        int need_block = UP_DIV(need_token, mBlockSize);
        // allocate the pointers to the page table
        for (int i = 0; i < need_block; ++i) {
            int slot_num = (need_token >= mBlockSize) ? mBlockSize : need_token;
            uint8_t* free_ptr = getFreePtr();
            StateCacheBlock* block = new StateCacheBlock({mCurrentReference->mRefId}, mBlockSize, slot_num);
            block->setTensors(shape);
            for (int s = 0; s < size.size(); ++s) {
                block->setTensorSize(s, size[s]);
                block->onAllocatePtr(free_ptr);
            }
            mCurrentReference->mPageTable[layer].push_back(std::shared_ptr(block));
            need_token -= mBlockSize;
        }
    } else {
        // 2.2 the naive one:
        // resize the only block and enlarge it!

    }
}

void StateCacheManager::prepareAttn(int ref_id, const std::vector<std::shared_ptr<StateCacheBlock>>& argv) {
    
}

} // namespace MNN