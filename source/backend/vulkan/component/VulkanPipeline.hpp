//
//  VulkanPipeline.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanPipeline_hpp
#define VulkanPipeline_hpp

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "NonCopyable.hpp"
#include "VulkanDevice.hpp"
#include "VulkanShaderMap.hpp"
#include "vulkan_wrapper.h"
namespace MNN {
class VulkanPipeline : public NonCopyable {
public:
    static VulkanPipeline* create(const VulkanDevice& dev, const uint8_t* data, size_t length,
                                  const std::vector<VkDescriptorType>& bufferTypes, VkPipelineCache cache,
                                  const std::vector<uint32_t>& localSize = std::vector<uint32_t>());
    virtual ~VulkanPipeline();

    VkPipeline get() const {
        return mPipeline;
    }

    void bind(VkCommandBuffer buffer, VkDescriptorSet describeSet) const;
    inline VkDescriptorType argType(int index) const {
        return mBufferTypes[index];
    }

    class DescriptorSet : public NonCopyable {
    public:
        DescriptorSet(const VulkanDevice& dev, VkDescriptorSet set, VkDescriptorPool pool,
                      const VulkanPipeline* pipeline)
            : mDevice(dev) {
            mSet      = set;
            mPool     = pool;
            mPipeline = pipeline;
        }
        virtual ~DescriptorSet() {
            mDevice.freeDescriptorSets(mPool, 1, &mSet);
            mDevice.destroyDescriptorPool(mPool);
        }

        void writeBuffer(VkBuffer buffer, int bindIndex, size_t size, VkDeviceSize offset = 0);
        void writeImage(VkImageView view, VkSampler sampler, VkImageLayout layout, int bind);

        VkDescriptorSet get() const {
            return mSet;
        }

    private:
        const VulkanDevice& mDevice;
        VkDescriptorSet mSet;
        VkDescriptorPool mPool;
        const VulkanPipeline* mPipeline;
    };

    DescriptorSet* createSet() const;

private:
    VulkanPipeline(const VulkanDevice& dev, VkPipeline p, VkPipelineLayout layout,
                   const std::vector<VkDescriptorPoolSize>& despool, VkDescriptorSetLayout setLayout,
                   const std::vector<VkDescriptorType>& bufferTypes);

    const VulkanDevice& mDevice;
    VkPipeline mPipeline;
    VkPipelineLayout mLayout;
    std::vector<VkDescriptorPoolSize> mDesPoolSize;
    VkDescriptorSetLayout mSetLayout;
    std::vector<VkDescriptorType> mBufferTypes;
};

class VulkanPipelineFactory : public NonCopyable {
public:
    VulkanPipelineFactory(const VulkanDevice& dev);
    ~VulkanPipelineFactory();
    const VulkanPipeline* getPipeline(const std::string& key, const std::vector<VkDescriptorType>& types,
                                      const std::vector<uint32_t>& localSize = std::vector<uint32_t>()) const;

private:
    const VulkanDevice& mDevice;
    mutable std::map<std::string, std::shared_ptr<VulkanPipeline>> mPipelines;
    VkPipelineCache mCache;

    std::shared_ptr<VulkanShaderMap> mStorage;
};
} // namespace MNN
#endif /* VulkanPipeline_hpp */
