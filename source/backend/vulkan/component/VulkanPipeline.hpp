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

class VulkanPipelineCache : public RefCount {
public:
    VulkanPipelineCache(const VulkanDevice& dev);
    virtual ~VulkanPipelineCache();
    inline VkPipelineCache get() const {
        return mCache;
    }
private:
    VkPipelineCache mCache;
    const VulkanDevice& mDevice;
};
class VulkanShaderModule : public RefCount {
public:
    static VulkanShaderModule* create(const VulkanDevice& dev, const uint32_t* buffer, size_t size);
    virtual ~ VulkanShaderModule();
    VkShaderModule get() const {
        return mShader;
    }
private:
    VulkanShaderModule(VkShaderModule shader, const VulkanDevice& dev);
    VkShaderModule mShader;
    const VulkanDevice& mDevice;
};

class VulkanLayout : public RefCount {
public:
    virtual ~ VulkanLayout();
    struct LayoutType {
        int binding;
        VkDescriptorType type;
        VkShaderStageFlagBits stage;
    };
    static VulkanLayout* create(const VulkanDevice& dev, const std::vector<LayoutType>& bufferTypes);
    friend class DescriptorSet;

    class DescriptorSet : public RefCount {
    public:
        DescriptorSet(VkDescriptorSet set, VkDescriptorPool pool,
                      const VulkanLayout* pipeline) {
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
        const VulkanLayout* mPipeline;
    };
    DescriptorSet* createSet() const;
    VkPipelineLayout get() const {
        return mLayout;
    }
private:
    std::vector<VkDescriptorType> mBufferTypes;
    VkPipelineLayout mLayout;
    std::vector<VkDescriptorPoolSize> mDesPoolSize;
    VkDescriptorSetLayout mSetLayout;
    const VulkanDevice& mDevice;

    VulkanLayout(const VulkanDevice& dev) : mDevice(dev) {
        // Do nothing
    }
};

class VulkanPipeline;
class VulkanGraphicPipelineCache : public RefCount {
public:
    struct ShaderSource {
        SharedPtr<VulkanShaderModule> vertex;
        SharedPtr<VulkanShaderModule> fragment;
    };
    static VulkanGraphicPipelineCache* create(const VulkanDevice& dev, const ShaderSource& source);
    virtual ~ VulkanGraphicPipelineCache();
    void setVertexFormats(const std::vector<int>& unit);
    
    // Complete info befor create pipeline //
    VkGraphicsPipelineCreateInfo& info() {
        /** Info content
        // Self or shader
        info.flags = 0;
        info.pStages;

        // Program layout or state
        info.pColorBlendState;
        info.pDepthStencilState;
        info.layout;
        info.pVertexInputState;
        info.pRasterizationState;

        // Drawable info
        info.pInputAssemblyState;
        
        // Render Pass Info
        info.pViewportState;
        info.pMultisampleState;
        
        // Render Pass Target
        info.renderPass;
         */
        return mInfo;
    }
private:
    VulkanGraphicPipelineCache(SharedPtr<VulkanShaderModule> vertex, SharedPtr<VulkanShaderModule> frag, const VulkanDevice& dev);
    SharedPtr<VulkanShaderModule> mVertex;
    SharedPtr<VulkanShaderModule> mFragment;
    const VulkanDevice& mDevice;
    VkGraphicsPipelineCreateInfo mInfo;
    VkPipelineVertexInputStateCreateInfo mVertexInfo;
    std::vector<VkVertexInputAttributeDescription> mVertexAttributes;
    std::vector<VkVertexInputBindingDescription> mVertexBindings;

    VkPipelineShaderStageCreateInfo mStage[2];
    VkPipelineColorBlendStateCreateInfo mBlend;
    VkPipelineDepthStencilStateCreateInfo mDepth;
    VkPipelineRasterizationStateCreateInfo mRasterization;
    VkPipelineInputAssemblyStateCreateInfo mInputAssembly;
    std::vector<VkPipelineColorBlendAttachmentState> mBlendAttchmentState;
    std::string mName;
};
class VulkanPipeline : public RefCount {
public:
    VulkanPipeline(const VulkanDevice& dev, VkPipeline p, SharedPtr<VulkanLayout> layout, VkPipelineBindPoint type, SharedPtr<VulkanShaderModule> shader, SharedPtr<VulkanPipelineCache> cache, const std::vector<uint32_t>& specConstants = std::vector<uint32_t>());
    virtual ~VulkanPipeline();

    VkPipeline get() const {
        return mPipeline;
    }
    VkPipelineLayout layout() const {
        return mLayout->get();
    }

    void bind(VkCommandBuffer buffer, VkDescriptorSet describeSet) const;
    VulkanLayout::DescriptorSet* createSet() const;
    void changePipeline(const std::vector<uint32_t>& localSize) const;

public:
    std::string mTuneName;

private:
    const VulkanDevice& mDevice;
    mutable VkPipeline mPipeline;
    VkPipelineBindPoint mType;
    SharedPtr<VulkanLayout> mLayout;
    SharedPtr<VulkanShaderModule> mShader;
    SharedPtr<VulkanPipelineCache> mCache;
    std::vector<uint32_t> mSpecConstants;
};

class VulkanPipelineFactory : public NonCopyable {
public:
    VulkanPipelineFactory(const VulkanDevice& dev);
    ~VulkanPipelineFactory();
    const VulkanPipeline* getPipeline(const std::string& key, const std::vector<VkDescriptorType>& types,
                                      const std::vector<uint32_t>& localSize = std::vector<uint32_t>(), 
                                      const std::vector<uint32_t>& specConstants = std::vector<uint32_t>(),
                                      const bool separate = false) const;
    SharedPtr<VulkanPipeline> getPrivatePipeline(const std::string& key, const std::vector<VkDescriptorType>& types, const std::vector<uint32_t>& specConstants = std::vector<uint32_t>());
    VulkanPipeline* createGraphicPipeline(SharedPtr<VulkanLayout> layout, VulkanGraphicPipelineCache* cache) const;
    VulkanPipeline* createComputePipeline(const uint8_t* data, size_t dataSize, const std::vector<VkDescriptorType>& types, 
                                          const std::vector<uint32_t>& localSize, 
                                          const std::vector<uint32_t>& specConstants = std::vector<uint32_t>()) const;
    SharedPtr<VulkanShaderModule> createShader(const std::string& key) const;
    void reset();
private:
    const VulkanDevice& mDevice;
    mutable std::map<std::string, SharedPtr<VulkanPipeline>> mPipelines;
    mutable std::map<const uint32_t *, SharedPtr<VulkanShaderModule>> mComputeShaderModules;
    SharedPtr<VulkanPipelineCache> mCache;
    std::shared_ptr<VulkanShaderMap> mStorage;
};
} // namespace MNN
#endif /* VulkanPipeline_hpp */
