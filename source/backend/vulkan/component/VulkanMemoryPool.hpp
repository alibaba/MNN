//
//  VulkanMemoryPool.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanMemoryPool_hpp
#define VulkanMemoryPool_hpp

#include <map>
#include <memory>
#include <vector>
#include "core/NonCopyable.hpp"
#include "backend/vulkan/component/VulkanDevice.hpp"
#include "backend/vulkan/vulkan/vulkan_wrapper.h"

namespace MNN {

class VulkanMemory : public NonCopyable {
public:
    VulkanMemory(const VulkanDevice& dev, const VkMemoryAllocateInfo& info);
    ~VulkanMemory();

    VkDeviceMemory get() const {
        return mMemory;
    }
    uint32_t type() const {
        return mTypeIndex;
    }
    VkDeviceSize size() const {
        return mSize;
    }

private:
    VkDeviceMemory mMemory;
    const VulkanDevice& mDevice;
    uint32_t mTypeIndex;
    VkDeviceSize mSize;
};

class VulkanMemoryPool : public NonCopyable {
public:
    VulkanMemoryPool(const VulkanDevice& dev, bool permitFp16);
    virtual ~VulkanMemoryPool();

    const VulkanMemory* allocMemory(const VkMemoryRequirements& requirements, VkFlags extraMask, bool seperate = false);
    void returnMemory(const VulkanMemory* memory, bool clean = false);

    // Free Unuseful Memory
    void clear();

    const VulkanDevice& device() const {
        return mDevice;
    }
    bool permitFp16() const {
        return mPermitFp16;
    }

    // Return MB
    float computeSize() const;

private:
    std::map<const VulkanMemory*, std::shared_ptr<VulkanMemory>> mAllBuffers;

    // MemoryTypeIndex, Size, Memory
    std::vector<std::multimap<VkDeviceSize, const VulkanMemory*>> mFreeBuffers;

    VkPhysicalDeviceMemoryProperties mPropty;
    const VulkanDevice& mDevice;
    bool mPermitFp16 = false;
};
} // namespace MNN
#endif /* VulkanMemoryPool_hpp */
