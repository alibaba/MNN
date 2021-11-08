//
//  VulkanSemaphore.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanSemaphore_hpp
#define VulkanSemaphore_hpp

#include <stdio.h>
#include <vector>
#include "core/NonCopyable.hpp"
#include "backend/vulkan/component/VulkanDevice.hpp"
#include "backend/vulkan/vulkan/vulkan_wrapper.h"

namespace MNN {
class VulkanSemaphore : public NonCopyable {
public:
    VulkanSemaphore(const VulkanDevice& dev);
    virtual ~VulkanSemaphore();

    VkSemaphore get() const {
        return mSemaphore;
    }

private:
    VkSemaphore mSemaphore;
    const VulkanDevice& mDevice;
};
} // namespace MNN
#endif /* VulkanSemaphore_hpp */
