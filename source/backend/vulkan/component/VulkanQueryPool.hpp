//
//  VulkanBasicExecution.hpp
//  MNN
//
//  Created by MNN on 2023/10/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanQueryPool_hpp
#define VulkanQueryPool_hpp

#include "core/NonCopyable.hpp"
#include "backend/vulkan/vulkan/vulkan_wrapper.h"
#include "VulkanDevice.hpp"

namespace MNN {

class VulkanQueryPool : public NonCopyable {
public:
    VulkanQueryPool(const VulkanDevice& dev);
    virtual ~VulkanQueryPool();

    void VulkanCmdResetQueryPool(VkCommandBuffer commandBuffer);
    void VulkanCmdWriteTimestamp(VkCommandBuffer commandBuffer, int index);
    float VulkanGetQueryPoolResults();

private:
    const VulkanDevice& mDevice;
    VkQueryPool queryPool;
};

} // namespace MNN
#endif  /* VulkanQueryPool_hpp */
