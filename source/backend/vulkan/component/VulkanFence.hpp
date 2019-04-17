//
//  VulkanFence.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanFence_hpp
#define VulkanFence_hpp

#include <stdio.h>
#include "NonCopyable.hpp"
#include "VulkanDevice.hpp"
#include "vulkan_wrapper.h"

// if support Fence FD ,force use FD Wait function, this macro only used for test purpose,
// if frameworks is blocked and not async , does not enable this macro
#define VK_FENCE_WAIT_FD_IF_SUPPORT (0)

namespace MNN {
class VulkanFence : public NonCopyable {
public:
    VulkanFence(const VulkanDevice& dev);
    virtual ~VulkanFence();

    VkFence get() const {
        return mFence;
    }
    VkResult reset() const;

    // if fenceFd() is called ,do'nt use this function wait, this function will blocked and never return
    // please call epoll_wait/select/poll wait for fd's signal
    VkResult wait() const;

    // if fenceFd is support, we can use epoll or select wait for fence complete
    bool supportFenceFd() const;

#ifdef VK_USE_PLATFORM_WIN32_KHR
    VkResult fenceFd(HANDLE& fd) const;
#else
    VkResult fenceFd(int& fd) const;
#endif

private:
    VkResult rawWait() const;

#if VK_FENCE_WAIT_FD_IF_SUPPORT
    VkResult fdWait() const;
    VkResult pollWait(const int fd) const;
#endif

private:
    VkFence mFence;
    const VulkanDevice& mDevice;
};
} // namespace MNN
#endif /* VulkanFence_hpp */
