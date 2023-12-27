//
//  OpenCLRunningUtils.cpp
//  MNN
//
//  Created by MNN on 2023/10/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanQueryPool.hpp"

namespace MNN {
VulkanQueryPool::VulkanQueryPool(const VulkanDevice& dev) : mDevice(dev){
    VkQueryPoolCreateInfo queryPoolCreateInfo = {};
    queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolCreateInfo.queryCount = 2; // 需要两个时间戳，一个用于开始，一个用于结束
    
    vkCreateQueryPool(mDevice.get(), &queryPoolCreateInfo, nullptr, &queryPool);
}

VulkanQueryPool::~VulkanQueryPool(){
    vkDestroyQueryPool(mDevice.get(), queryPool, nullptr);
}

void VulkanQueryPool::VulkanCmdResetQueryPool(VkCommandBuffer commandBuffer){
    vkCmdResetQueryPool(commandBuffer, queryPool, 0, 2);
}

void VulkanQueryPool::VulkanCmdWriteTimestamp(VkCommandBuffer commandBuffer, int index){
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, index);
}

float VulkanQueryPool::VulkanGetQueryPoolResults(){
    uint64_t timestamps[2];
    vkGetQueryPoolResults(mDevice.get(), queryPool, 0, 2, sizeof(uint64_t) * 2, timestamps, sizeof(uint64_t), VK_QUERY_RESULT_WAIT_BIT);
    
    float timestampPeriod = mDevice.getTimestampPeriod();
    float executionTime = (timestamps[1] - timestamps[0]) * timestampPeriod * 1e-3f; // 微妙
    return executionTime;
}
} // namespace MNN
