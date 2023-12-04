#ifndef VulkanFramebuffer_hpp
#define VulkanFramebuffer_hpp
#include "VulkanDevice.hpp"
#include "core/AutoStorage.h"
namespace MNN {
class VulkanFramebuffer : public RefCount {
public:
    virtual ~ VulkanFramebuffer();
    
    static VulkanFramebuffer* create(const VulkanDevice& dev, const VkFramebufferCreateInfo* info);
    inline VkFramebuffer get() const {
        return mContent;
    }
private:
    VulkanFramebuffer(const VulkanDevice& dev, VkFramebuffer fram) : mDevice(dev) {
        mContent = fram;
    }
    VkFramebuffer mContent;
    const VulkanDevice& mDevice;
};
};
#endif
