//
//  VulkanPipeline.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanPipeline.hpp"
#include <string.h>
#include <map>
namespace MNN {
VulkanPipelineFactory::VulkanPipelineFactory(const VulkanDevice& dev) : mDevice(dev) {
    CALL_VK(mDevice.createPipelineCache(mCache));
    mStorage = std::make_shared<VulkanShaderMap>();
}
VulkanPipelineFactory::~VulkanPipelineFactory() {
    mDevice.destroyPipelineCache(mCache);
}

VulkanPipeline::VulkanPipeline(const VulkanDevice& dev, VkPipeline p, VkPipelineLayout layout,
                               const std::vector<VkDescriptorPoolSize>& despool, VkDescriptorSetLayout setLayout,
                               const std::vector<VkDescriptorType>& bufferTypes)
    : mDevice(dev) {
    mPipeline    = p;
    mLayout      = layout;
    mDesPoolSize = despool;
    mSetLayout   = setLayout;
    mBufferTypes = bufferTypes;
}

VulkanPipeline* VulkanPipeline::create(const VulkanDevice& dev, const uint8_t* data, size_t length,
                                       const std::vector<VkDescriptorType>& bufferTypes, VkPipelineCache cache,
                                       const std::vector<uint32_t>& localSize) {
    VkShaderModule shaderOut;
    VkResult result = dev.createShaderModule(shaderOut, length, (const uint32_t*)data);
    if (VK_SUCCESS != result) {
        return nullptr;
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::map<VkDescriptorType, int> typeCount;
    for (int i = 0; i < bufferTypes.size(); ++i) {
        auto type = bufferTypes[i];
        if (typeCount.find(type) == typeCount.end()) {
            typeCount[type] = 1;
        } else {
            typeCount[type] += 1;
        }
        VkDescriptorSetLayoutBinding binding;
        binding.binding            = i;
        binding.descriptorType     = type;
        binding.descriptorCount    = 1;
        binding.stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
        binding.pImmutableSamplers = nullptr;
        bindings.emplace_back(binding);
    }

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    {
        CALL_VK(dev.createDescriptorSetLayout(setLayout, bindings.size(), bindings.data()));
        CALL_VK(dev.createPipelineLayout(pipelineLayout, setLayout));
    }

    /*for localSize_x_id = 0,localSize_y_id = 1,localSize_z_id = 2*/
    std::vector<VkSpecializationMapEntry> specializationMapEntry; /*localSize data description*/
    std::shared_ptr<VkSpecializationInfo> specializationInfo = std::make_shared<VkSpecializationInfo>();
    if (localSize.size() > 0) {
        // FUNC_PRINT(localSize.size());
        for (int i = 0; i < localSize.size(); i++) {
            VkSpecializationMapEntry entry = {(uint32_t)(i + 1), (uint32_t)(sizeof(uint32_t) * i),
                                              sizeof(uint32_t)}; /*id,offset,length*/
            specializationMapEntry.push_back(entry);
        }
        specializationInfo->pData         = localSize.data();
        specializationInfo->dataSize      = localSize.size() * sizeof(uint32_t); /*bytes*/
        specializationInfo->pMapEntries   = specializationMapEntry.data();
        specializationInfo->mapEntryCount = specializationMapEntry.size();
    }

    // Create the pipeline cache
    VkPipeline pipeline;
    auto res = dev.createComputePipeline(pipeline, shaderOut, pipelineLayout, cache, specializationInfo.get());
    if (VK_SUCCESS != res) {
        FUNC_PRINT(res);
        dev.destroyShaderModule(shaderOut);
        dev.destroyPipelineLayout(pipelineLayout);
        dev.destroyDescriptorSetLayout(setLayout);
        return nullptr;
    }
    dev.destroyShaderModule(shaderOut);

    std::vector<VkDescriptorPoolSize> desPoolSize;
    for (auto& iter : typeCount) {
        VkDescriptorPoolSize s;
        s.descriptorCount = iter.second;
        s.type            = iter.first;
        desPoolSize.emplace_back(s);
    }

    return new VulkanPipeline(dev, pipeline, pipelineLayout, desPoolSize, setLayout, bufferTypes);
}
VulkanPipeline::~VulkanPipeline() {
    mDevice.destroyPipelineLayout(mLayout);
    mDevice.destroyDescriptorSetLayout(mSetLayout);
    mDevice.destroyPipeline(mPipeline);
    // FUNC_PRINT(1);
}

void VulkanPipeline::bind(VkCommandBuffer cmd, VkDescriptorSet des) const {
    // Bind the compute pipeline.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mPipeline);
    // Bind descriptor set.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mLayout, 0, 1, &des, 0, nullptr);
}

VulkanPipeline::DescriptorSet* VulkanPipeline::createSet() const {
    VkDescriptorPool descriptorPool;
    //        FUNC_PRINT(poolInfo.poolSizeCount);
    CALL_VK(mDevice.createDescriptorPool(descriptorPool, mDesPoolSize.size(), mDesPoolSize.data()));

    VkDescriptorSet descriptorSet;
    CALL_VK(mDevice.allocateDescriptorSet(descriptorSet, descriptorPool, mSetLayout));
    return new DescriptorSet(mDevice, descriptorSet, descriptorPool, this);
}

void VulkanPipeline::DescriptorSet::writeBuffer(VkBuffer buffer, int bindIndex, size_t size, VkDeviceSize offset) {
    VkWriteDescriptorSet writeSet;
    ::memset(&writeSet, 0, sizeof(writeSet));
    VkDescriptorBufferInfo sourceInfo;
    sourceInfo.buffer        = buffer;
    sourceInfo.offset        = offset;
    sourceInfo.range         = size;
    writeSet.descriptorCount = 1;
    writeSet.descriptorType  = mPipeline->argType(bindIndex);
    writeSet.dstBinding      = bindIndex;
    writeSet.pBufferInfo     = &sourceInfo;
    writeSet.dstSet          = mSet;

    mDevice.updateWriteDescriptorSet(writeSet);
}

void VulkanPipeline::DescriptorSet::writeImage(VkImageView view, VkSampler sampler, VkImageLayout layout, int bind) {
    VkWriteDescriptorSet writeSet;
    ::memset(&writeSet, 0, sizeof(writeSet));
    VkDescriptorImageInfo sourceInfo;
    sourceInfo.imageView   = view;
    sourceInfo.imageLayout = layout;
    sourceInfo.sampler     = sampler;

    writeSet.descriptorCount = 1;
    writeSet.descriptorType  = mPipeline->argType(bind);
    writeSet.dstBinding      = bind;
    writeSet.pImageInfo      = &sourceInfo;
    writeSet.dstSet          = mSet;

    mDevice.updateWriteDescriptorSet(writeSet);
}

const VulkanPipeline* VulkanPipelineFactory::getPipeline(const std::string& key,
                                                         const std::vector<VkDescriptorType>& types,
                                                         const std::vector<uint32_t>& localSize) const {
    auto iter = mPipelines.find(key);
    if (iter != mPipelines.end()) {
        return iter->second.get();
    }

    auto content = mStorage->search(key);
    if (nullptr == content.first) {
        MNN_ERROR("Don't find shader for %s\n", key.c_str());
        return nullptr;
    }
    auto pipeline = VulkanPipeline::create(mDevice, content.first, content.second, types, mCache, localSize);
    if (nullptr != pipeline) {
        mPipelines.insert(std::make_pair(key, std::shared_ptr<VulkanPipeline>(pipeline)));
    } else {
        MNN_ERROR("Error for create pipeline %s\n", key.c_str());
    }
    return pipeline;
}

} // namespace MNN
