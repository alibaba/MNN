//
//  VulkanBuffer.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanBuffer_hpp
#define VulkanBuffer_hpp
#include "VulkanMemoryPool.hpp"
namespace MNN {
class VulkanBuffer : public NonCopyable {
public:
    VulkanBuffer(const VulkanMemoryPool& pool, bool separate, size_t size, const void* hostData = nullptr,
                 VkBufferUsageFlags usage  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VkSharingMode shared      = VK_SHARING_MODE_EXCLUSIVE,
                 VkFlags requirements_mask = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    virtual ~VulkanBuffer();

    VkBuffer buffer() const {
        return mBuffer;
    }
    size_t size() const {
        return mSize;
    }
    void* map(int start = 0, int size = -1) const;
    void unmap() const;

    void flush(bool write, int start, int size) const;

    void release();

private:
    const VulkanMemoryPool& mPool;
    MemChunk mMemory;
    VkBuffer mBuffer;
    size_t mSize;
    VkBufferUsageFlags mUsage;
    bool mReleased = false;
    VkSharingMode mShared;
};
} // namespace MNN

#endif /* VulkanBuffer_hpp */
