//
//  VulkanBuffer.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanBuffer.hpp"
#include <string.h>
namespace MNN {

VulkanBuffer::VulkanBuffer(const VulkanMemoryPool& pool, bool seperate, size_t size, const void* hostData,
                           VkBufferUsageFlags usage, VkSharingMode shared, VkFlags requirements_mask)
    : mPool(pool) {
    MNN_ASSERT(size > 0);
    mSize = size;
    CALL_VK(mPool.device().createBuffer(mBuffer, mSize, usage, shared));

    VkMemoryRequirements memReq;
    mPool.device().getBufferMemoryRequirements(mBuffer, memReq);
    mMemory = const_cast<VulkanMemoryPool&>(mPool).allocMemory(memReq, requirements_mask, seperate);
    //        FUNC_PRINT(mMemory->type());

    if (nullptr != hostData) {
        void* data = nullptr;
        CALL_VK(mPool.device().mapMemory(mMemory->get(), 0, size, 0, &data));
        ::memcpy(data, hostData, size);
        mPool.device().unmapMemory(mMemory->get());
    }

    CALL_VK(mPool.device().bindBufferMemory(mBuffer, mMemory->get()));
}

VulkanBuffer::~VulkanBuffer() {
    mPool.device().destroyBuffer(mBuffer);
    if (!mReleased) {
        const_cast<VulkanMemoryPool&>(mPool).returnMemory(mMemory, true);
    }
}
void* VulkanBuffer::map(int start, int size) const {
    if (size < 0) {
        size = mSize;
    }
    void* data = nullptr;
    CALL_VK(mPool.device().mapMemory(mMemory->get(), start, size, 0, &data));
    return data;
}
void VulkanBuffer::unmap() const {
    mPool.device().unmapMemory(mMemory->get());
}
void VulkanBuffer::release() {
    if (mReleased) {
        return;
    }
    mReleased = true;
    const_cast<VulkanMemoryPool&>(mPool).returnMemory(mMemory);
}

void VulkanBuffer::flush(bool write, int start, int size) const {
    VkMappedMemoryRange range;
    range.memory = mMemory->get();
    range.offset = start;
    range.size   = size;
    range.pNext  = nullptr;

    if (write) {
        CALL_VK(mPool.device().flushMappedMemoryRanges(&range));
    } else {
        CALL_VK(mPool.device().invalidateMappedMemoryRanges(&range));
    }
}

} // namespace MNN
