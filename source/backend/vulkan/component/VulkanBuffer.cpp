//
//  VulkanBuffer.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/component/VulkanBuffer.hpp"
#include <string.h>
namespace MNN {

VulkanBuffer::VulkanBuffer(const VulkanMemoryPool& pool, bool separate, size_t size, const void* hostData,
                           VkBufferUsageFlags usage, VkSharingMode shared, VkFlags requirements_mask)
    : mPool(pool) {
    MNN_ASSERT(size > 0);
    mSize = size;
    mShared = shared;
    mBuffer = const_cast<VulkanMemoryPool&>(mPool).allocBuffer(size, usage, shared);
    mUsage = usage;
    if (mBuffer == VK_NULL_HANDLE) {
        return;
    }

    VkMemoryRequirements memReq;
    mPool.device().getBufferMemoryRequirements(mBuffer, memReq);
    mMemory = const_cast<VulkanMemoryPool&>(mPool).allocMemory(memReq, requirements_mask, separate);
    if (nullptr == mMemory.first) {
        MNN_ERROR("VulkanBuffer: allocMemory failed, size=%zu\n", size);
        return;
    }
    auto realMem = (VulkanMemory*)mMemory.first;

    if (mPool.device().bindBufferMemory(mBuffer, realMem->get(), mMemory.second) != VK_SUCCESS) {
        MNN_ERROR("VulkanBuffer: bindBufferMemory failed, size=%zu\n", size);
        return;
    }
    mBound = true;
    if (nullptr != hostData) {
        void* data = nullptr;
        if (mPool.device().mapMemory(realMem->get(), mMemory.second, size, 0, &data) != VK_SUCCESS) {
            mBound = false;
            return;
        }
        ::memcpy(data, hostData, size);
        mPool.device().unmapMemory(realMem->get());
    }
}

VulkanBuffer::~VulkanBuffer() {
    if (mBuffer != VK_NULL_HANDLE) {
        const_cast<VulkanMemoryPool&>(mPool).returnBuffer(mBuffer, mSize, mUsage, mShared);
    }
    if (!mReleased && mMemory.first) {
        const_cast<VulkanMemoryPool&>(mPool).returnMemory(mMemory);
    }
}
void* VulkanBuffer::map(int start, int size) const {
    if (!mBound)
        return nullptr;
    if (size < 0) {
        size = mSize;
    }
    auto realMem = (VulkanMemory*)mMemory.first;
    void* data = nullptr;
    if (mPool.device().mapMemory(realMem->get(), start + mMemory.second, size, 0, &data) != VK_SUCCESS) {
        return nullptr;
    }
    return data;
}
void VulkanBuffer::unmap() const {
    auto realMem = (VulkanMemory*)mMemory.first;
    mPool.device().unmapMemory(realMem->get());
}
void VulkanBuffer::release() {
    if (mReleased || !mMemory.first) {
        return;
    }
    mReleased = true;
    const_cast<VulkanMemoryPool&>(mPool).returnMemory(mMemory);
}

void VulkanBuffer::flush(bool write, int start, int size) const {
    // Do nothing
}

} // namespace MNN
