#ifndef VulkanRenderPass_hpp
#define VulkanRenderPass_hpp
#include "core/AutoStorage.h"
#include "VulkanDevice.hpp"
namespace MNN {
class VulkanRenderPass : public RefCount {
public:
    virtual ~ VulkanRenderPass();
    
    VkRenderPass get() const {
        return mPass;
    }
    static VulkanRenderPass* create(const VulkanDevice& dev, const VkRenderPassCreateInfo* info);
private:
    VulkanRenderPass(VkRenderPass pass, const VulkanDevice& dev);
    VkRenderPass mPass;
    const VulkanDevice& mDevice;
};

};
#endif
