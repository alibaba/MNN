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

static VkFormat _getFormat(halide_type_t type) {
    switch (type.code) {
        case halide_type_float:
            return VK_FORMAT_R32G32B32A32_SFLOAT;
        case halide_type_int: {
            if (8 == type.bits) {
                return VK_FORMAT_R8G8B8A8_SINT;
            } else if (type.bits == 16) {
                return VK_FORMAT_R16G16B16A16_SINT;
            }
            return VK_FORMAT_R32G32B32A32_SINT;
        }
        case halide_type_uint: {
            if (8 == type.bits) {
                return VK_FORMAT_R8G8B8A8_UINT;
            } else if (type.bits == 16) {
                return VK_FORMAT_R16G16B16A16_UINT;
            }
            return VK_FORMAT_R32G32B32A32_UINT;
        }
        default:
            break;
    }
    return VK_FORMAT_R32G32B32A32_SFLOAT;
}

VulkanImage::VulkanImage(const VulkanMemoryPool& pool, bool seperate, const std::vector<int>& dims, halide_type_t type)
    : mDevice(pool.device()), mPool(pool) {
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

    auto format = _getFormat(type);
    if (pool.permitFp16() && format == VK_FORMAT_R32G32B32A32_SFLOAT) {
        // Use fp16 instead of fp32
        format = VK_FORMAT_R16G16B16A16_SFLOAT;
    }
    auto mFormat     = format;
    mInfo = std::make_tuple(imageType, mWidth, mHeight, mDepth, mFormat);
    // FUNC_PRINT(format);
    mImage.first = const_cast<VulkanMemoryPool&>(mPool).allocImage(mInfo);
    mLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkMemoryRequirements memRequirements;
    mDevice.getImageMemoryRequirements(mImage.first, memRequirements);

    mMemory = const_cast<VulkanMemoryPool&>(mPool).allocMemory(memRequirements, 0, seperate);
    //        FUNC_PRINT(mMemory->type());
    auto realMem = (VulkanMemory*)mMemory.first;
    mDevice.bindImageMemory(mImage.first, realMem->get(), mMemory.second);
    CALL_VK(mDevice.createImageView(mImage.second, mImage.first, viewType, format));
}
VulkanImage::~VulkanImage() {
    mDevice.destroyImageView(mImage.second, nullptr);
    const_cast<VulkanMemoryPool&>(mPool).returnImage(std::move(mImage.first), std::move(mInfo));
    if (nullptr != mMemory.first) {
        const_cast<VulkanMemoryPool&>(mPool).returnMemory(mMemory);
    }
}

} // namespace MNN
