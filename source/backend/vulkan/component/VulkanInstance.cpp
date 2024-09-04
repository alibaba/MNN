//
//  VulkanInstance.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/component/VulkanInstance.hpp"
#include <vector>
#include <algorithm>

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

    // Set instance extensions.
    std::vector<const char*> instanceExtensions;
    std::vector<const char*> instanceExtensionsToCheck = {
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    };
    uint32_t availableInstanceExtensionCount = 0;
    CALL_VK(vkEnumerateInstanceExtensionProperties(nullptr, &availableInstanceExtensionCount, nullptr));
    std::vector<VkExtensionProperties> availableInstanceExtensions(availableInstanceExtensionCount);
    CALL_VK(vkEnumerateInstanceExtensionProperties(nullptr, &availableInstanceExtensionCount, availableInstanceExtensions.data()));
    for (uint32_t i = 0; i < availableInstanceExtensionCount; i++) {
        for (uint32_t j = 0; j < instanceExtensionsToCheck.size(); j++) {
            if (strcmp(availableInstanceExtensions[i].extensionName, instanceExtensionsToCheck[j]) == 0) {
                instanceExtensions.push_back(instanceExtensionsToCheck[j]);
            }
        }
    }

    // Set instanceCreateFlag.
    auto it = std::find_if(instanceExtensions.begin(), instanceExtensions.end(),
                        [](const char* str) { return strcmp(str, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0; });
    VkInstanceCreateFlags instanceCreateFlag = (it != instanceExtensions.end()) ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0;

#ifdef MNN_VULKAN_DEBUG
    MNN_PRINT("MNN_VULKAN_DEBUG is on.\n");
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };
#endif

    // Create the Vulkan instance
    VkInstanceCreateInfo instanceCreateInfo{
        /* .sType                   = */ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        /* .pNext                   = */ nullptr,
        /* .flags                   = */ instanceCreateFlag,
        /* .pApplicationInfo        = */ &appInfo,
#ifdef MNN_VULKAN_DEBUG
        /* .enabledLayerCount       = */ 1,
        /* .ppEnabledLayerNames     = */ validationLayers.data(),
#else
        /* .enabledLayerCount       = */ 0,
        /* .ppEnabledLayerNames     = */ nullptr,
#endif
        /* .enabledExtensionCount   = */ static_cast<uint32_t>(instanceExtensions.size()),
        /* .ppEnabledExtensionNames = */ instanceExtensions.data(),
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
    if (VK_NULL_HANDLE == mInstance) {
        return false;
    }
    uint32_t gpuCount = 0;
    auto res          = enumeratePhysicalDevices(gpuCount, nullptr);
    if ((0 == gpuCount) || (VK_SUCCESS != res)) {
        MNN_ERROR("Invalide device for support vulkan\n");
        return false;
    }
    return true;
}
} // namespace MNN
