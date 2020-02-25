//
//  VulkanFence.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/component/VulkanFence.hpp"
#if VK_FENCE_WAIT_FD_IF_SUPPORT
#include <errno.h>
#include <poll.h>
#include <string.h>
#include <MNN/MNNDefine.h>
#endif

namespace MNN {
VulkanFence::VulkanFence(const VulkanDevice& dev) : mDevice(dev) {
    CALL_VK(mDevice.createFence(mFence));
}
VulkanFence::~VulkanFence() {
    mDevice.destroyFence(mFence);
}

VkResult VulkanFence::rawWait() const {
    auto status = VK_TIMEOUT;
    do {
        status = mDevice.waitForFence(mFence, 5000000000);
    } while (status == VK_TIMEOUT);
    return status;
}

VkResult VulkanFence::wait() const {
    return rawWait();
}
VkResult VulkanFence::reset() const {
    return mDevice.resetFence(mFence);
}
} // namespace MNN
