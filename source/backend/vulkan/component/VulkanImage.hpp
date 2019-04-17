//
//  VulkanImage.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanImage_hpp
#define VulkanImage_hpp
#include "Tensor.hpp"
#include "VulkanBuffer.hpp"
#include "VulkanMemoryPool.hpp"
namespace MNN {
class VulkanSampler : public NonCopyable {
public:
    VulkanSampler(const VulkanDevice& dev, VkFilter filter = VK_FILTER_NEAREST,
                  VkSamplerAddressMode mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
    virtual ~VulkanSampler();

    VkSampler get() const {
        return mSampler;
    }

private:
    VkSampler mSampler;
    const VulkanDevice& mDevice;
};

class VulkanImage : public NonCopyable {
public:
    VulkanImage(const VulkanMemoryPool& pool, bool seperate, const std::vector<int>& dims,
                halide_type_t type = halide_type_of<float>());
    VulkanImage(const VulkanMemoryPool& pool, bool seperate, int w, int h)
        : VulkanImage(pool, seperate, std::vector<int>{w, h}) {
    }
    virtual ~VulkanImage();

    inline int width() const {
        return mWidth;
    }
    inline int height() const {
        return mHeight;
    }
    inline int depth() const {
        return mDepth;
    }
    inline std::vector<int> dims() const {
        return mDims;
    }
    inline VkImage get() const {
        return mImage;
    }
    inline VkImageView view() const {
        return mImageView;
    }
    inline VkFormat format() const {
        return mFormat;
    }

    void release();

private:
    VkImage mImage;
    VkImageView mImageView;
    VkFormat mFormat;
    const VulkanDevice& mDevice;
    int mWidth;
    int mHeight;
    int mDepth;

    std::vector<int> mDims;
    const VulkanMemoryPool& mPool;
    const VulkanMemory* mMemory;
    bool mReleased = false;
};
} // namespace MNN

#endif /* VulkanImage_hpp */
