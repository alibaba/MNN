//
//  VulkanCommandPool.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/component/VulkanCommandPool.hpp"
#include <string.h>
#include <memory>
#include "backend/vulkan/component/VulkanFence.hpp"
#include "backend/vulkan/component/VulkanImage.hpp"

namespace MNN {
VulkanCommandPool::VulkanCommandPool(const VulkanDevice& dev) : mDevice(dev), mPool(VK_NULL_HANDLE) {
    CALL_VK(mDevice.createCommandPool(mPool));
    MNN_ASSERT(VK_NULL_HANDLE != mPool);
}
VulkanCommandPool::~VulkanCommandPool() {
    for (auto& b : mFreeBuffers) {
        mDevice.freeCommandBuffer(mPool, b);
    }
    mDevice.destroyCommandPool(mPool);
    // FUNC_PRINT(1);
}

void VulkanCommandPool::submitAndWait(VkCommandBuffer buffer) const {
    auto b                   = buffer;
    auto fence               = std::make_shared<VulkanFence>(mDevice);
    VkSubmitInfo submit_info = {/* .sType                = */ VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                /* .pNext                = */ nullptr,
                                /* .waitSemaphoreCount   = */ 0,
                                /* .pWaitSemaphores      = */ nullptr,
                                /* .pWaitDstStageMask    = */ nullptr,
                                /* .commandBufferCount   = */ 1,
                                /* .pCommandBuffers      = */ &b,
                                /* .signalSemaphoreCount = */ 0,
                                /* .pSignalSemaphores    = */ nullptr};
    auto fenceReal           = fence->get();
    auto queue               = mDevice.acquireDefaultDevQueue();
    CALL_VK(vkQueueSubmit(queue, 1, &submit_info, fenceReal));
    fence->wait();
}

VulkanCommandPool::Buffer* VulkanCommandPool::allocBuffer() const {
    return new Buffer(this);
}

VulkanCommandPool::Buffer::Buffer(const VulkanCommandPool* pool) : mPool(pool) {
    if (pool->mFreeBuffers.empty()) {
        CALL_VK(pool->mDevice.allocateCommandBuffer(pool->mPool, mBuffer));
    } else {
        auto iter = pool->mFreeBuffers.end() - 1;
        mBuffer = *iter;
        pool->mFreeBuffers.erase(iter);
    }
}
VulkanCommandPool::Buffer::~Buffer() {
    mPool->mFreeBuffers.emplace_back(mBuffer);
}
void VulkanCommandPool::Buffer::barrierSource(std::tuple<VkBuffer, VkDeviceSize, VkDeviceSize> fuse, BarrierType type) const {
    barrierSource(std::get<0>(fuse), std::get<2>(fuse), std::get<1>(fuse), type);
}

void VulkanCommandPool::Buffer::barrierSource(VkBuffer source, size_t start, size_t size, BarrierType type) const {
    VkBufferMemoryBarrier barrier;
    barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.buffer              = source;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.offset              = start;
    barrier.pNext               = nullptr;
    barrier.size                = size;
    switch (type) {
        case READ_WRITE:
            barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            break;
        case WRITE_WRITE:
            barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            break;
        case WRITE_READ:
            barrier.srcAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            break;
        default:
            break;
    }
    vkCmdPipelineBarrier(mBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
}
void VulkanCommandPool::Buffer::begin(VkCommandBufferUsageFlags flag) const {
    VkCommandBufferBeginInfo cmdBufferBeginInfo{
        /* .sType            = */ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        /* .pNext            = */ nullptr,
        /* .flags            = */ flag,
        /* .pInheritanceInfo = */ nullptr,
    };
    vkResetCommandBuffer(mBuffer, 0);
    CALL_VK(vkBeginCommandBuffer(mBuffer, &cmdBufferBeginInfo));
}
void VulkanCommandPool::Buffer::end() const {
    CALL_VK(vkEndCommandBuffer(mBuffer));
}

} // namespace MNN
