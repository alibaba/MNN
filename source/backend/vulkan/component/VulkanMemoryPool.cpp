//
//  VulkanMemoryPool.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanMemoryPool.hpp"
namespace MNN {
VulkanMemory::VulkanMemory(const VulkanDevice& dev, const VkMemoryAllocateInfo& info) : mDevice(dev) {
    CALL_VK(mDevice.allocMemory(mMemory, info));
    mTypeIndex = info.memoryTypeIndex;
    mSize      = info.allocationSize;
}
VulkanMemory::~VulkanMemory() {
    mDevice.freeMemory(mMemory);
}

VulkanMemoryPool::VulkanMemoryPool(const VulkanDevice& dev) : mDevice(dev) {
    mDevice.getPhysicalDeviceMemoryProperties(mPropty);
    mFreeBuffers.resize(mPropty.memoryTypeCount);
}
VulkanMemoryPool::~VulkanMemoryPool() {
}

const VulkanMemory* VulkanMemoryPool::allocMemory(const VkMemoryRequirements& requirements, VkFlags extraMask,
                                                  bool seperate) {
    uint32_t index = 0;
    auto typeBits  = requirements.memoryTypeBits;
    for (uint32_t i = 0; i < mPropty.memoryTypeCount; i++) {
        if ((typeBits & 1) == 1) {
            // Type is available, does it match user properties?
            if ((mPropty.memoryTypes[i].propertyFlags & extraMask) == extraMask) {
                index = i;
                break;
            }
        }
        typeBits >>= 1;
    }
    MNN_ASSERT(index >= 0);
    if (!seperate) {
        auto freeIter = mFreeBuffers[index].lower_bound(requirements.size);
        if (freeIter != mFreeBuffers[index].end()) {
            auto result = freeIter->second;
            mFreeBuffers[index].erase(freeIter);
            return result;
        }
    }

    VkMemoryAllocateInfo allocInfo{
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext           = nullptr,
        .allocationSize  = requirements.size,
        .memoryTypeIndex = index, // Memory type assigned in the next step
    };

    auto memory = std::make_shared<VulkanMemory>(mDevice, allocInfo);
    mAllBuffers.insert(std::make_pair(memory.get(), memory));
    return memory.get();
}

void VulkanMemoryPool::returnMemory(const VulkanMemory* memory, bool clean) {
    if (!clean) {
        mFreeBuffers[memory->type()].insert(std::make_pair(memory->size(), memory));
        return;
    }
    auto iter = mAllBuffers.find(memory);
    if (iter != mAllBuffers.end()) {
        mAllBuffers.erase(iter);
    }
    return;
}

void VulkanMemoryPool::clear() {
    for (auto& iter : mFreeBuffers) {
        for (auto& subIter : iter) {
            auto eraseIter = mAllBuffers.find(subIter.second);
            if (eraseIter != mAllBuffers.end()) {
                mAllBuffers.erase(eraseIter);
            }
        }
        iter.clear();
    }
}

float VulkanMemoryPool::computeSize() const {
    float totalSize = 0;
    for (auto& iter : mAllBuffers) {
        totalSize += (float)(iter.first->size());
    }
    return totalSize / 1024.0f / 1024.0f;
}
} // namespace MNN
