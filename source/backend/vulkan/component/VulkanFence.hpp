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
#include "core/NonCopyable.hpp"
#include "backend/vulkan/component/VulkanDevice.hpp"
#include "backend/vulkan/vulkan/vulkan_wrapper.h"

// if support Fence FD ,force use FD Wait function, this macro only used for test purpose,
// if frameworks is blocked and not async , does not enable this macro

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

private:
    VkResult rawWait() const;

private:
    VkFence mFence;
    const VulkanDevice& mDevice;
};
} // namespace MNN
#endif /* VulkanFence_hpp */
