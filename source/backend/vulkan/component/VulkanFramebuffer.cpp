#include "VulkanFramebuffer.hpp"
namespace MNN {
VulkanFramebuffer::~VulkanFramebuffer() {
    vkDestroyFramebuffer(mDevice.get(), mContent, nullptr);
}

VulkanFramebuffer* VulkanFramebuffer::create(const VulkanDevice& dev, const VkFramebufferCreateInfo* fbCreateInfo) {
    VkFramebuffer frame;
    auto res = vkCreateFramebuffer(dev.get(), fbCreateInfo, nullptr,
                                &frame);
    if (VK_SUCCESS != res) {
        MNN_ERROR("Create Vulkan Framebuffer error\n");
        return nullptr;
    }
    return new VulkanFramebuffer(dev, frame);
}

};
