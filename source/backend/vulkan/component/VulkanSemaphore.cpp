//
//  VulkanSemaphore.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/component/VulkanSemaphore.hpp"
namespace MNN {
VulkanSemaphore::VulkanSemaphore(const VulkanDevice& dev) : mDevice(dev) {
    CALL_VK(mDevice.createSemaphore(mSemaphore));
}
VulkanSemaphore::~VulkanSemaphore() {
    mDevice.destroySemaphore(mSemaphore);
}

} // namespace MNN
