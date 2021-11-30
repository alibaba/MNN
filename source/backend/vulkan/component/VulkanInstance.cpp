//
//  VulkanInstance.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/component/VulkanInstance.hpp"
#include <vector>

namespace MNN {
VulkanInstance::VulkanInstance() : mOwner(true), mInstance(VK_NULL_HANDLE) {
    VkApplicationInfo appInfo = {
        /* .sType              = */ VK_STRUCTURE_TYPE_APPLICATION_INFO,
        /* .pNext              = */ nullptr,
        /* .pApplicationName   = */ "MNN_Vulkan",
        /* .applicationVersion = */ VK_MAKE_VERSION(1, 0, 0),
        /* .pEngineName        = */ "Compute",
        /* .engineVersion      = */ VK_MAKE_VERSION(1, 0, 0),
        /* .apiVersion         = */ VK_MAKE_VERSION(1, 0, 0),
    };
    std::vector<const char*> instance_extensions;
#ifdef MNN_VULKAN_DEBUG
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };
#endif
    // Create the Vulkan instance
    VkInstanceCreateInfo instanceCreateInfo{
        /* .sType                   = */ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        /* .pNext                   = */ nullptr,
        /* .flags                   = */ 0,
        /* .pApplicationInfo        = */ &appInfo,
#ifdef MNN_VULKAN_DEBUG
        /* .enabledLayerCount       = */ 1,
        /* .ppEnabledLayerNames     = */ validationLayers.data(),
#else
        /* .enabledLayerCount       = */ 0,
        /* .ppEnabledLayerNames     = */ nullptr,
#endif
        /* .enabledExtensionCount   = */ static_cast<uint32_t>(instance_extensions.size()),
        /* .ppEnabledExtensionNames = */ instance_extensions.data(),
    };
    CALL_VK(vkCreateInstance(&instanceCreateInfo, nullptr, &mInstance));
}
VulkanInstance::VulkanInstance(VkInstance instance) : mOwner(false), mInstance(instance) {
}

VulkanInstance::~VulkanInstance() {
    if (mOwner && (VK_NULL_HANDLE != mInstance)) {
        vkDestroyInstance(mInstance, nullptr);
        mInstance = VK_NULL_HANDLE;
    }
}
const VkResult VulkanInstance::enumeratePhysicalDevices(uint32_t& physicalDeviceCount,
                                                        VkPhysicalDevice* physicalDevices) const {
    return vkEnumeratePhysicalDevices(get(), &physicalDeviceCount, physicalDevices);
}

void VulkanInstance::getPhysicalDeviceQueueFamilyProperties(const VkPhysicalDevice& physicalDevice,
                                                            uint32_t& queueFamilyPropertyCount,
                                                            VkQueueFamilyProperties* pQueueFamilyProperties) {
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, pQueueFamilyProperties);
}

const bool VulkanInstance::supportVulkan() const {
    uint32_t gpuCount = 0;
    auto res          = enumeratePhysicalDevices(gpuCount, nullptr);
    if ((0 == gpuCount) || (VK_SUCCESS != res)) {
        MNN_ERROR("Invalide device for support vulkan\n");
        return false;
    }
    return true;
}
} // namespace MNN
