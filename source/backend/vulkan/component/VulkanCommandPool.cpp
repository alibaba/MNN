//
//  VulkanCommandPool.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanCommandPool.hpp"
#include <string.h>
#include <memory>
#include "VulkanFence.hpp"
namespace MNN {
VulkanCommandPool::VulkanCommandPool(const VulkanDevice& dev) : mDevice(dev), mPool(VK_NULL_HANDLE) {
    CALL_VK(mDevice.createCommandPool(mPool));
    MNN_ASSERT(VK_NULL_HANDLE != mPool);
}
VulkanCommandPool::~VulkanCommandPool() {
    mDevice.destroyCommandPool(mPool);
    // FUNC_PRINT(1);
}

void VulkanCommandPool::submitAndWait(VkCommandBuffer buffer) const {
    auto b                   = buffer;
    auto fence               = std::make_shared<VulkanFence>(mDevice);
    VkSubmitInfo submit_info = {.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                .pNext                = nullptr,
                                .waitSemaphoreCount   = 0,
                                .pWaitSemaphores      = nullptr,
                                .pWaitDstStageMask    = nullptr,
                                .commandBufferCount   = 1,
                                .pCommandBuffers      = &b,
                                .signalSemaphoreCount = 0,
                                .pSignalSemaphores    = nullptr};
    auto fenceReal           = fence->get();
    auto queue               = mDevice.acquireDefaultDevQueue();
    CALL_VK(vkQueueSubmit(queue, 1, &submit_info, fenceReal));
    fence->wait();
}

const VulkanCommandPool::Buffer* VulkanCommandPool::allocBuffer() const {
    return new Buffer(mPool, mDevice);
}

VulkanCommandPool::Buffer::Buffer(const VkCommandPool& pool, const VulkanDevice& dev) : mPool(pool), mDevice(dev) {
    CALL_VK(mDevice.allocateCommandBuffer(mPool, mBuffer));
}
VulkanCommandPool::Buffer::~Buffer() {
    mDevice.freeCommandBuffer(mPool, mBuffer);
}

void VulkanCommandPool::Buffer::barrierImage(VkImage source, VkImageLayout oldLayout, VkImageLayout newLayout) const {
    VkImageMemoryBarrier barrier;
    ::memset(&barrier, 0, sizeof(VkImageMemoryBarrier));

    barrier.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
    barrier.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstAccessMask               = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcAccessMask               = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.image                       = source;
    barrier.newLayout                   = newLayout;
    barrier.oldLayout                   = oldLayout;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(mBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &barrier);
}

void VulkanCommandPool::Buffer::barrierSource(VkBuffer source, size_t start, size_t size) const {
    VkBufferMemoryBarrier barrier;
    barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.buffer              = source;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.offset              = start;
    barrier.pNext               = nullptr;
    barrier.size                = size;
    barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(mBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
}

void VulkanCommandPool::Buffer::begin(VkCommandBufferUsageFlags flag) const {
    VkCommandBufferBeginInfo cmdBufferBeginInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = flag,
        .pInheritanceInfo = nullptr,
    };
    vkResetCommandBuffer(mBuffer, 0);
    CALL_VK(vkBeginCommandBuffer(mBuffer, &cmdBufferBeginInfo));
}
void VulkanCommandPool::Buffer::end() const {
    CALL_VK(vkEndCommandBuffer(mBuffer));
}

} // namespace MNN
