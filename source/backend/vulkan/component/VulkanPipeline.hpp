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
#include "VulkanDevice.hpp"
#include "VulkanShaderMap.hpp"
#include "core/AutoStorage.h"
namespace MNN {
class VulkanPipeline : public RefCount {
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
        DescriptorSet(VkDescriptorSet set, VkDescriptorPool pool,
                      const VulkanPipeline* pipeline) {
            mSet      = set;
            mPool     = pool;
            mPipeline = pipeline;
        }
        virtual ~DescriptorSet();

        void writeBuffer(VkBuffer buffer, int bindIndex, size_t size, VkDeviceSize offset = 0);
        void writeBuffer(std::tuple<VkBuffer, VkDeviceSize, VkDeviceSize> fuseBuffer, int bindIndex);

        void writeImage(VkImageView view, VkSampler sampler, VkImageLayout layout, int bind);

        VkDescriptorSet get() const {
            return mSet;
        }

    private:
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
    mutable std::vector<std::pair<VkDescriptorSet, VkDescriptorPool>> mFreeSets;
};

class VulkanPipelineFactory : public NonCopyable {
public:
    VulkanPipelineFactory(const VulkanDevice& dev);
    ~VulkanPipelineFactory();
    const VulkanPipeline* getPipeline(const std::string& key, const std::vector<VkDescriptorType>& types,
                                      const std::vector<uint32_t>& localSize = std::vector<uint32_t>()) const;

    void reset();
private:
    const VulkanDevice& mDevice;
    mutable std::map<std::string, SharedPtr<VulkanPipeline>> mPipelines;
    VkPipelineCache mCache;

    std::shared_ptr<VulkanShaderMap> mStorage;
};
} // namespace MNN
#endif /* VulkanPipeline_hpp */
