//
//  VulkanCommandPool.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanCommandPool_hpp
#define VulkanCommandPool_hpp

#include "core/NonCopyable.hpp"
#include "backend/vulkan/component/VulkanDevice.hpp"
#include "backend/vulkan/vulkan/vulkan_wrapper.h"
namespace MNN {
class VulkanImage;
class VulkanCommandPool : public NonCopyable {
public:
    VulkanCommandPool(const VulkanDevice& dev);
    virtual ~VulkanCommandPool();

    class Buffer : public NonCopyable {
    public:
        Buffer(const VulkanCommandPool* pool);
        virtual ~Buffer();

        VkCommandBuffer get() const {
            return mBuffer;
        }

        void begin(VkCommandBufferUsageFlags flags) const;
        void end() const;
        enum BarrierType {
            READ_WRITE = 0,
            WRITE_WRITE,
            WRITE_READ,
        };
        void barrierSource(VkBuffer source, size_t start, size_t end, BarrierType type = READ_WRITE) const;
        void barrierSource(std::tuple<VkBuffer, VkDeviceSize, VkDeviceSize>, BarrierType type = READ_WRITE) const;
    private:
        VkCommandBuffer mBuffer;
        const VulkanCommandPool* mPool;
    };

    VulkanCommandPool::Buffer* allocBuffer() const;

    VkCommandPool pool() const {
        return mPool;
    }

    void submitAndWait(VkCommandBuffer buffer) const;

private:
    const VulkanDevice& mDevice;
    VkCommandPool mPool;
    mutable std::vector<VkCommandBuffer> mFreeBuffers;
};
} // namespace MNN
#endif /* VulkanCommandPool_hpp */
