//
//  VulkanImage.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanImage.hpp"
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
    if (mReleased) {
        return;
    }
    mReleased = true;
    const_cast<VulkanMemoryPool&>(mPool).returnMemory(mMemory);
}

static VkFormat _getFormat(halide_type_t type) {
    switch (type.code) {
        case halide_type_float:
            return VK_FORMAT_R16G16B16A16_SFLOAT;
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
    return VK_FORMAT_R16G16B16A16_SFLOAT;
}

VulkanImage::VulkanImage(const VulkanMemoryPool& pool, bool seperate, const std::vector<int>& dims, halide_type_t type)
    : mPool(pool), mDevice(pool.device()) {
    MNN_ASSERT(dims.size() >= 1 && dims.size() <= 3);
    auto imageType = VK_IMAGE_TYPE_1D;
    auto viewType  = VK_IMAGE_VIEW_TYPE_1D;
    mDims          = dims;
    mWidth         = dims[0];
    mHeight        = 1;
    mDepth         = 1;
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
    mFormat     = format;
    // FUNC_PRINT(format);
    CALL_VK(mDevice.createImage(mImage, imageType, mWidth, mHeight, mDepth, format));

    VkMemoryRequirements memRequirements;
    mDevice.getImageMemoryRequirements(mImage, memRequirements);

    mMemory = const_cast<VulkanMemoryPool&>(mPool).allocMemory(memRequirements, 0, seperate);
    //        FUNC_PRINT(mMemory->type());

    mDevice.bindImageMemory(mImage, mMemory->get());

    CALL_VK(mDevice.createImageView(mImageView, mImage, viewType, format));
}
VulkanImage::~VulkanImage() {
    mDevice.destroyImageView(mImageView);
    mDevice.destroyImage(mImage);
    if (!mReleased) {
        const_cast<VulkanMemoryPool&>(mPool).returnMemory(mMemory, true);
    }
}

} // namespace MNN
