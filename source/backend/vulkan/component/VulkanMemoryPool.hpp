//
//  VulkanMemoryPool.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanMemoryPool_hpp
#define VulkanMemoryPool_hpp

#include <map>
#include <memory>
#include <vector>
#include "core/NonCopyable.hpp"
#include "component/VulkanDevice.hpp"
#include "vulkan/vulkan_wrapper.h"
#include "core/BufferAllocator.hpp"

namespace MNN {

class VulkanMemory : public NonCopyable {
public:
    VulkanMemory(const VulkanDevice& dev, const VkMemoryAllocateInfo& info);
    ~VulkanMemory();

    VkDeviceMemory get() const {
        return mMemory;
    }
    uint32_t type() const {
        return mTypeIndex;
    }
    VkDeviceSize size() const {
        return mSize;
    }

private:
    VkDeviceMemory mMemory;
    const VulkanDevice& mDevice;
    uint32_t mTypeIndex;
    VkDeviceSize mSize;
};

class VulkanMemoryPool : public NonCopyable {
public:
    VulkanMemoryPool(const VulkanDevice& dev, bool permitFp16);
    VulkanMemoryPool(const VulkanMemoryPool* parent);
    virtual ~VulkanMemoryPool();

    // VulkanMemory* , offset
    MemChunk allocMemory(const VkMemoryRequirements& requirements, VkFlags extraMask, bool separate = false);
    void returnMemory(MemChunk memory);

    // Free Unuseful Memory
    void clear();

    const VulkanDevice& device() const {
        return mDevice;
    }
    bool permitFp16() const {
        return mPermitFp16;
    }

    // Return MB
    float computeSize() const;

    // For buffer fast alloc
    VkBuffer allocBuffer(size_t size, VkBufferUsageFlags flags, VkSharingMode shared);
    void returnBuffer(VkBuffer buffer, size_t size, VkBufferUsageFlags flags, VkSharingMode shared);

private:
    // MemoryTypeIndex, Size, Memory
    std::vector<std::shared_ptr<BufferAllocator>> mAllocators;

    const VulkanDevice& mDevice;
    bool mPermitFp16 = false;
};
} // namespace MNN
#endif /* VulkanMemoryPool_hpp */
