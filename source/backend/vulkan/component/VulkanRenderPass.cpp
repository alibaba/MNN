#include "VulkanRenderPass.hpp"
namespace MNN {
VulkanRenderPass::VulkanRenderPass(VkRenderPass pass, const VulkanDevice& dev) : mDevice(dev) {
    mPass = pass;
}
VulkanRenderPass::~VulkanRenderPass() {
    vkDestroyRenderPass(mDevice.get(), mPass, nullptr);
}

VulkanRenderPass* VulkanRenderPass::create(const VulkanDevice& dev, const VkRenderPassCreateInfo* info) {
    VkRenderPass pass;
    auto res = vkCreateRenderPass(dev.get(), info, nullptr, &pass);
    if (VK_SUCCESS != res) {
        MNN_ERROR("Create Vulkan Render Pass error\n");
        return nullptr;
    }
    return new VulkanRenderPass(pass, dev);
}

};
