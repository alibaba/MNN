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

VulkanMemoryPool::VulkanMemoryPool(const VulkanDevice& dev, bool permitFp16) : mDevice(dev) {
    mDevice.getPhysicalDeviceMemoryProperties(mPropty);
    mFreeBuffers.resize(mPropty.memoryTypeCount);
    mPermitFp16 = permitFp16;
}
VulkanMemoryPool::~VulkanMemoryPool() {
    clear();
}

VkBuffer VulkanMemoryPool::allocBuffer(size_t size, VkBufferUsageFlags flags, VkSharingMode shared) {
    auto iter = mFreeVkBuffers.find(std::make_tuple(size, flags, shared));
    if (iter != mFreeVkBuffers.end()) {
        auto res = iter->second;
        mFreeVkBuffers.erase(iter);
        return res;
    }
    VkBuffer res;
    CALL_VK(mDevice.createBuffer(res, size, flags, shared));
    return res;
}

void VulkanMemoryPool::returnBuffer(VkBuffer buffer, size_t size, VkBufferUsageFlags flags, VkSharingMode shared) {
    mFreeVkBuffers.insert(std::make_pair(std::make_tuple(size, flags, shared), buffer));
}

std::shared_ptr<VulkanMemory> VulkanMemoryPool::allocMemory(const VkMemoryRequirements& requirements, VkFlags extraMask,
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
    MNN_ASSERT(index < mFreeBuffers.size());
    auto freeIter = mFreeBuffers[index].lower_bound(requirements.size);
    if (!seperate) {
        if (freeIter != mFreeBuffers[index].end()) {
            auto result = freeIter->second;
            mFreeBuffers[index].erase(freeIter);
            return result;
        }
    } else {
        // For debug
        //FUNC_PRINT(index);
    }
    VkMemoryAllocateInfo allocInfo{
        /* .sType           = */ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        /* .pNext           = */ nullptr,
        /* .allocationSize  = */ requirements.size,
        /* .memoryTypeIndex = */ index, // Memory type assigned in the next step
    };

    auto memory = std::make_shared<VulkanMemory>(mDevice, allocInfo);
    mAllocedSize += memory->size() / 1024.0f / 1024.0f;
    return memory;
}

void VulkanMemoryPool::returnMemory(std::shared_ptr<VulkanMemory> memory) {
    mFreeBuffers[memory->type()].insert(std::make_pair(memory->size(), memory));
    mAllocedSize -= memory->size() / 1024.0f / 1024.0f;
    return;
}

void VulkanMemoryPool::clear() {
    for (auto& iter : mFreeBuffers) {
        iter.clear();
    }
    for (auto& iter : mFreeVkBuffers) {
        mDevice.destroyBuffer(iter.second);
    }
    mFreeVkBuffers.clear();
    for (auto& iter : mFreeImages) {
        mDevice.destroyImage(iter.second);
    }
    mFreeImages.clear();
}
VkImage VulkanMemoryPool::allocImage(const std::tuple<VkImageType, uint32_t, uint32_t, uint32_t, VkFormat>& info) {
    auto iter = mFreeImages.find(info);
    if (iter != mFreeImages.end()) {
        auto res = iter->second;
        mFreeImages.erase(iter);
        return res;
    }
    VkImage image;
    VkImageView imageView;
    CALL_VK(mDevice.createImage(image, std::get<0>(info), std::get<1>(info), std::get<2>(info), std::get<3>(info), std::get<4>(info)));
    return image;
}
void VulkanMemoryPool::returnImage(VkImage dst, std::tuple<VkImageType, uint32_t, uint32_t, uint32_t, VkFormat>&& info) {
    mFreeImages.insert(std::make_pair(info, dst));
}


float VulkanMemoryPool::computeSize() const {
    float totalSize = 0;
    for (auto& piter : mFreeBuffers) {
        for (auto& iter : piter) {
            totalSize += (float)(iter.first);
        }
    }
    return totalSize / 1024.0f / 1024.0f + mAllocedSize;
}
} // namespace MNN
