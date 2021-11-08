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

class VulkanAllocator : public BufferAllocator::Allocator {
public:
    VulkanAllocator(const VulkanDevice& device, int index) : mDevice(device), mIndex(index) {
        // Do nothing
    }
    virtual ~ VulkanAllocator() {
        // Do nothing
    }
    virtual std::pair<void*, int> onAlloc(int size) override {
        VkMemoryAllocateInfo info;
        info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        info.pNext = nullptr;
        info.allocationSize = size;
        info.memoryTypeIndex = mIndex;
        auto mem = new VulkanMemory(mDevice, info);
        return std::make_pair(mem, 0);
    }
    virtual void onRelease(std::pair<void*, int> ptr) override {
        auto p = (VulkanMemory*)ptr.first;
        delete p;
    }
private:
    const VulkanDevice& mDevice;
    int mIndex;
};

VulkanMemoryPool::VulkanMemoryPool(const VulkanDevice& dev, bool permitFp16) : mDevice(dev) {
    mAllocators.resize(dev.memProty().memoryTypeCount);
    for (int i=0; i<mAllocators.size(); ++i) {
        std::shared_ptr<BufferAllocator::Allocator> allocReal(new VulkanAllocator(dev, i));
        mAllocators[i].reset(new BufferAllocator(allocReal, dev.proty().limits.nonCoherentAtomSize));
    }
    mPermitFp16 = permitFp16;
}
VulkanMemoryPool::VulkanMemoryPool(const VulkanMemoryPool* parent) : mDevice(parent->mDevice) {
    mPermitFp16 = parent->mPermitFp16;
    mAllocators.resize(mDevice.memProty().memoryTypeCount);
    for (int i=0; i<mAllocators.size(); ++i) {
        std::shared_ptr<BufferAllocator::Allocator> allocReal = BufferAllocator::Allocator::createRecurse(parent->mAllocators[i].get());
        mAllocators[i].reset(new BufferAllocator(allocReal, parent->mDevice.proty().limits.nonCoherentAtomSize));
    }
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

std::pair<void*, int> VulkanMemoryPool::allocMemory(const VkMemoryRequirements& requirements, VkFlags extraMask,
                                                  bool seperate) {
    uint32_t index = 0;
    auto typeBits  = requirements.memoryTypeBits;
    for (uint32_t i = 0; i < mDevice.memProty().memoryTypeCount; i++) {
        if ((typeBits & 1) == 1) {
            // Type is available, does it match user properties?
            if ((mDevice.memProty().memoryTypes[i].propertyFlags & extraMask) == extraMask) {
                index = i;
                break;
            }
        }
        typeBits >>= 1;
    }
    MNN_ASSERT(index >= 0);
    MNN_ASSERT(index < mAllocators.size());
    return mAllocators[index]->alloc(requirements.size, seperate);
}

void VulkanMemoryPool::returnMemory(std::pair<void*, int> memory) {
    auto mem = (VulkanMemory*)memory.first;
    mAllocators[mem->type()]->free(memory);
    return;
}

void VulkanMemoryPool::clear() {
    for (auto& iter : mAllocators) {
        iter->release(false);
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
    for (auto& piter : mAllocators) {
        totalSize += (float)piter->totalSize();
    }
    return totalSize / 1024.0f / 1024.0f;
}
} // namespace MNN
