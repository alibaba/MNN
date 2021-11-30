//
//  VulkanInstance.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanInstance_hpp
#define VulkanInstance_hpp

#include "core/NonCopyable.hpp"
#include "backend/vulkan/vulkan/vulkan_wrapper.h"
namespace MNN {
class VulkanInstance : public NonCopyable {
public:
    VulkanInstance();
    explicit VulkanInstance(VkInstance instance);
    virtual ~VulkanInstance();

    const VkResult enumeratePhysicalDevices(uint32_t& physicalDeviceCount, VkPhysicalDevice* pPhysicalDevices) const;
    void getPhysicalDeviceQueueFamilyProperties(const VkPhysicalDevice& physicalDevice,
                                                uint32_t& queueFamilyPropertyCount,
                                                VkQueueFamilyProperties* pQueueFamilyProperties);

    const bool supportVulkan() const;

    VkInstance get() const {
        return mInstance;
    }

    bool success() const {
        return (VK_NULL_HANDLE != mInstance);
    }

private:
    bool mOwner;
    VkInstance mInstance;
};
} // namespace MNN
#endif /* VulkanInstance_hpp */
