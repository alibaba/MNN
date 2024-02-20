//
//  VulkanImage.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/component/VulkanImage.hpp"
#include <string.h>
namespace MNN {
VulkanSampler::VulkanSampler(const VulkanDevice& dev, VkFilter filter, VkSamplerAddressMode mode) : mDevice(dev) {
    // Finally, create a sampler.
    CALL_VK(mDevice.createSampler(mSampler, filter, mode));
}
VulkanSampler::~VulkanSampler() {
    mDevice.destroySampler(mSampler);
}
void VulkanImage::release() {
    if (nullptr == mMemory.first) {
        return;
    }
    const_cast<VulkanMemoryPool&>(mPool).returnMemory(mMemory);
    mMemory.first = nullptr;
}

VulkanImage::VulkanImage(const VulkanMemoryPool& pool, bool separate, const std::vector<int>& dims, VkFormat format, VkImageUsageFlags usage)
    : mDevice(pool.device()), mPool(pool) {
    if (format == VK_FORMAT_R32G32B32A32_SFLOAT && pool.permitFp16()) {
        // FIXME: find better method
        format = VK_FORMAT_R16G16B16A16_SFLOAT;
    }
    MNN_ASSERT(dims.size() >= 1 && dims.size() <= 3);
    auto imageType = VK_IMAGE_TYPE_1D;
    auto viewType  = VK_IMAGE_VIEW_TYPE_1D;
    mDims          = dims;
    auto mWidth         = dims[0];
    auto mHeight        = 1;
    auto mDepth         = 1;
    if (dims.size() > 1) {
        mHeight   = dims[1];
        imageType = VK_IMAGE_TYPE_2D;
        viewType  = VK_IMAGE_VIEW_TYPE_2D;
    }
    if (dims.size() > 2) {
        mDepth    = dims[2];
        imageType = VK_IMAGE_TYPE_3D;
        viewType  = VK_IMAGE_VIEW_TYPE_3D;
    }

    auto mFormat     = format;
    mInfo = std::make_tuple(imageType, mWidth, mHeight, mDepth, mFormat);
    // FUNC_PRINT(format);
    CALL_VK(mDevice.createImage(mImage.first, imageType, mWidth, mHeight, mDepth, mFormat, usage));
    VkMemoryRequirements memRequirements;
    mDevice.getImageMemoryRequirements(mImage.first, memRequirements);

    mMemory = const_cast<VulkanMemoryPool&>(mPool).allocMemory(memRequirements, 0, separate);
    //        FUNC_PRINT(mMemory->type());
    auto realMem = (VulkanMemory*)mMemory.first;
    mDevice.bindImageMemory(mImage.first, realMem->get(), mMemory.second);
    CALL_VK(mDevice.createImageView(mImage.second, mImage.first, viewType, format));
}
VulkanImage::~VulkanImage() {
    mDevice.destroyImageView(mImage.second, nullptr);
    mDevice.destroyImage(mImage.first);
    if (nullptr != mMemory.first) {
        const_cast<VulkanMemoryPool&>(mPool).returnMemory(mMemory);
    }
}
void VulkanImage::barrierWrite(VkCommandBuffer buffer) const {
    VkImageSubresourceRange subrange;
    ::memset(&subrange, 0, sizeof(VkImageSubresourceRange));
    subrange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subrange.layerCount = 1;
    subrange.levelCount = 1;
    insertMemoryBarrier(buffer, mImage.first, mAccess, VK_ACCESS_SHADER_WRITE_BIT, mLayout, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, subrange);
    mLayout = VK_IMAGE_LAYOUT_GENERAL;
    mAccess = VK_ACCESS_SHADER_WRITE_BIT;
}
void VulkanImage::barrierRead(VkCommandBuffer buffer) const {
    if (mAccess == VK_ACCESS_SHADER_READ_BIT && mLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        return;
    }
    VkImageSubresourceRange subrange;
    ::memset(&subrange, 0, sizeof(VkImageSubresourceRange));
    subrange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subrange.layerCount = 1;
    subrange.levelCount = 1;
    insertMemoryBarrier(buffer, mImage.first, mAccess, VK_ACCESS_SHADER_READ_BIT, mLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, subrange);
    mLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    mAccess = VK_ACCESS_SHADER_READ_BIT;
}

void VulkanImage::insertMemoryBarrier(
    VkCommandBuffer cmdbuffer,
    VkImage image,
    VkAccessFlags srcAccessMask,
    VkAccessFlags dstAccessMask,
    VkImageLayout oldImageLayout,
    VkImageLayout newImageLayout,
    VkPipelineStageFlags srcStageMask,
    VkPipelineStageFlags dstStageMask,
    VkImageSubresourceRange subresourceRange
                                      ) {
    VkImageMemoryBarrier imageMemoryBarrier;
    ::memset(&imageMemoryBarrier, 0, sizeof(VkImageMemoryBarrier));
    imageMemoryBarrier.srcAccessMask = srcAccessMask;
    imageMemoryBarrier.dstAccessMask = dstAccessMask;
    imageMemoryBarrier.oldLayout = oldImageLayout;
    imageMemoryBarrier.newLayout = newImageLayout;
    imageMemoryBarrier.image = image;
    imageMemoryBarrier.subresourceRange = subresourceRange;

    vkCmdPipelineBarrier(
        cmdbuffer,
        srcStageMask,
        dstStageMask,
        0,
        0, nullptr,
        0, nullptr,
        1, &imageMemoryBarrier);
}


} // namespace MNN
