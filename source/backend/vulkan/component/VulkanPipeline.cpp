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
    mCache = new VulkanPipelineCache(dev);
    mStorage = std::make_shared<VulkanShaderMap>();
}
VulkanPipelineFactory::~VulkanPipelineFactory() {
}

void VulkanPipelineFactory::reset() {
    mCache = nullptr;
    mCache = new VulkanPipelineCache(mDevice);
}

SharedPtr<VulkanShaderModule> VulkanPipelineFactory::createShader(const std::string& key) const {
    auto content = mStorage->search(key);
    if (nullptr == content.first) {
        MNN_ERROR("Don't find shader for %s\n", key.c_str());
        return nullptr;
    }
    SharedPtr<VulkanShaderModule> shader = VulkanShaderModule::create(mDevice, (const uint32_t*)content.first, content.second);
    return shader;
}

VulkanPipeline* VulkanPipelineFactory::createGraphicPipeline(SharedPtr<VulkanLayout> layout, VulkanGraphicPipelineCache* cache) const {
    VkPipeline pipeline;
    auto& mInfo = cache->info();
    mInfo.layout = layout->get();
    auto res = vkCreateGraphicsPipelines(mDevice.get(), mCache->get(), 1, &mInfo, nullptr, &pipeline);
    if (VK_SUCCESS != res) {
        MNN_ERROR("Create Graphic pipeline error: %d\n", res);
        return nullptr;
    }
    return new VulkanPipeline(mDevice, pipeline, layout, VK_PIPELINE_BIND_POINT_GRAPHICS, nullptr, mCache);
}
VulkanPipeline* VulkanPipelineFactory::createComputePipeline(const uint8_t* data, size_t dataSize, const std::vector<VkDescriptorType>& types, const std::vector<uint32_t>& localSize, const std::vector<uint32_t>& specConstants) const {
    SharedPtr<VulkanShaderModule> shader;
    auto iter = mComputeShaderModules.find((const uint32_t*)data);
    if (iter == mComputeShaderModules.end()) {
        shader = VulkanShaderModule::create(mDevice, (const uint32_t*)data, dataSize);
        mComputeShaderModules.insert(std::make_pair((const uint32_t*)data, shader));
    } else {
        shader = iter->second;
    }
    std::vector<VulkanLayout::LayoutType> layoutTypes(types.size());
    for (int i=0; i<types.size(); ++i) {
        layoutTypes[i].binding = i;
        layoutTypes[i].type = types[i];
        layoutTypes[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    SharedPtr<VulkanLayout> layout = VulkanLayout::create(mDevice, layoutTypes);
    VkPipeline pipeline;
    /*for localSize_x_id = 0,localSize_y_id = 1,localSize_z_id = 2*/
    std::vector<VkSpecializationMapEntry> specializationMapEntry; /*localSize data description*/
    std::shared_ptr<VkSpecializationInfo> specializationInfo;
    std::vector<uint32_t> totalSpecData;
    if (localSize.size() > 0 || specConstants.size() > 0) {
        MNN_ASSERT(localSize.size() <= 3);
        specializationInfo = std::make_shared<VkSpecializationInfo>();
        totalSpecData = localSize;
        totalSpecData.insert(totalSpecData.end(), specConstants.begin(), specConstants.end());
        // FUNC_PRINT(localSize.size());
        for (int i = 0; i < localSize.size(); i++) {
            VkSpecializationMapEntry entry = {(uint32_t)(i), (uint32_t)(sizeof(uint32_t) * i),
                                              sizeof(uint32_t)}; /*id,offset,length*/
            specializationMapEntry.push_back(entry);
        }
        for (int i = 0; i < specConstants.size(); i++) {
            VkSpecializationMapEntry entry = {(uint32_t)(3 + i), (uint32_t)(sizeof(uint32_t) * (localSize.size() + i)),
                                              sizeof(uint32_t)}; /*id,offset,length*/
            specializationMapEntry.push_back(entry);
        }
        specializationInfo->pData         = totalSpecData.data();
        specializationInfo->dataSize      = totalSpecData.size() * sizeof(uint32_t); /*bytes*/
        specializationInfo->pMapEntries   = specializationMapEntry.data();
        specializationInfo->mapEntryCount = specializationMapEntry.size();
    }
    auto res = mDevice.createComputePipeline(pipeline, shader->get(), layout->get(), mCache->get(), specializationInfo.get());
    if (VK_SUCCESS != res) {
        FUNC_PRINT(res);
        return nullptr;
    }
    return new VulkanPipeline(mDevice, pipeline, layout, VK_PIPELINE_BIND_POINT_COMPUTE, shader, mCache, specConstants);
}

const VulkanPipeline* VulkanPipelineFactory::getPipeline(const std::string& key,
                                                         const std::vector<VkDescriptorType>& types,
                                                         const std::vector<uint32_t>& localSize,
                                                         const std::vector<uint32_t>& specConstants,
                                                         const bool separate) const {
    std::string pipelineKey = key;
    for(int i = 0; i < localSize.size(); ++i){
        pipelineKey += "_" + std::to_string(localSize[i]);
    }
    for(int i = 0; i < specConstants.size(); ++i){
        pipelineKey += "_spec" + std::to_string(specConstants[i]);
    }
    if(separate){
        pipelineKey += "_tuned";
    }
    auto iter = mPipelines.find(pipelineKey);
    if (iter != mPipelines.end()) {
        return iter->second.get();
    }
    auto content = mStorage->search(key);
    if (nullptr == content.first) {
        MNN_ERROR("Don't find shader for %s\n", key.c_str());
        return nullptr;
    }
    auto pipeline = createComputePipeline((uint8_t*)content.first, content.second, types, localSize, specConstants);
    SharedPtr<VulkanPipeline> resPipeline = pipeline;
    mPipelines.insert(std::make_pair(pipelineKey, resPipeline));
    return pipeline;
}

SharedPtr<VulkanPipeline> VulkanPipelineFactory::getPrivatePipeline(const std::string& key, const std::vector<VkDescriptorType>& types, const std::vector<uint32_t>& specConstants) {
    std::pair<const unsigned char*, size_t> content = mStorage->search(key);
    if (nullptr == content.first) {
        MNN_ERROR("Don't find shader for %s\n", key.c_str());
        return nullptr;
    }

    VulkanPipeline * pipeline = createComputePipeline((uint8_t*)content.first, content.second, types, {}, specConstants);
    pipeline->mTuneName = key;
    SharedPtr<VulkanPipeline> resPipeline = pipeline;
    return resPipeline;
}

VulkanPipeline::VulkanPipeline(const VulkanDevice& dev, VkPipeline p, SharedPtr<VulkanLayout> layout, VkPipelineBindPoint type, SharedPtr<VulkanShaderModule> shader, SharedPtr<VulkanPipelineCache> cache, const std::vector<uint32_t>& specConstants)
    : mDevice(dev) {
    mPipeline    = p;
    mLayout      = layout;
    mType = type;
    mShader = shader;
    mCache = cache;
    mSpecConstants = specConstants;
}

VulkanPipeline::~VulkanPipeline() {
    mDevice.destroyPipeline(mPipeline);
    // FUNC_PRINT(1);
}

void VulkanPipeline::bind(VkCommandBuffer cmd, VkDescriptorSet des) const {
    // Bind the compute pipeline.
    vkCmdBindPipeline(cmd, mType, mPipeline);
    // Bind descriptor set.
    vkCmdBindDescriptorSets(cmd, mType, mLayout->get(), 0, 1, &des, 0, nullptr);
}
VulkanLayout::DescriptorSet* VulkanPipeline::createSet() const {
    return mLayout->createSet();
}

void VulkanPipeline::changePipeline(const std::vector<uint32_t>& localSize) const{
    mDevice.destroyPipeline(mPipeline);
    /*for localSize_x_id = 0,localSize_y_id = 1,localSize_z_id = 2*/
    std::vector<VkSpecializationMapEntry> specializationMapEntry; /*localSize data description*/
    std::shared_ptr<VkSpecializationInfo> specializationInfo = std::make_shared<VkSpecializationInfo>();
    std::vector<uint32_t> totalSpecData;
    if (localSize.size() > 0 || mSpecConstants.size() > 0) {
        totalSpecData = localSize;
        totalSpecData.insert(totalSpecData.end(), mSpecConstants.begin(), mSpecConstants.end());
        // FUNC_PRINT(localSize.size());
        for (int i = 0; i < localSize.size(); i++) {
            VkSpecializationMapEntry entry = {(uint32_t)(i), (uint32_t)(sizeof(uint32_t) * i),
                                              sizeof(uint32_t)}; /*id,offset,length*/
            specializationMapEntry.push_back(entry);
        }
        for (int i = 0; i < mSpecConstants.size(); i++) {
            VkSpecializationMapEntry entry = {(uint32_t)(3 + i), (uint32_t)(sizeof(uint32_t) * (localSize.size() + i)),
                                              sizeof(uint32_t)}; /*id,offset,length*/
            specializationMapEntry.push_back(entry);
        }
        specializationInfo->pData         = totalSpecData.data();
        specializationInfo->dataSize      = totalSpecData.size() * sizeof(uint32_t); /*bytes*/
        specializationInfo->pMapEntries   = specializationMapEntry.data();
        specializationInfo->mapEntryCount = specializationMapEntry.size();
    }
    
    auto res = mDevice.createComputePipeline(mPipeline, mShader->get(), mLayout->get(), mCache->get(), specializationInfo.get());
    if (VK_SUCCESS != res) {
        FUNC_PRINT(1);
    }
}

VulkanLayout::DescriptorSet* VulkanLayout::createSet() const {
    VkDescriptorPool descriptorPool;
    //        FUNC_PRINT(poolInfo.poolSizeCount);
    CALL_VK(mDevice.createDescriptorPool(descriptorPool, mDesPoolSize.size(), mDesPoolSize.data()));

    VkDescriptorSet descriptorSet;
    CALL_VK(mDevice.allocateDescriptorSet(descriptorSet, descriptorPool, mSetLayout));
    return new DescriptorSet(descriptorSet, descriptorPool, this);
}

VulkanLayout::DescriptorSet::~DescriptorSet() {
    mPipeline->mDevice.freeDescriptorSets(mPool, 1, &mSet);
    mPipeline->mDevice.destroyDescriptorPool(mPool);
}
void VulkanLayout::DescriptorSet::writeBuffer(std::tuple<VkBuffer, VkDeviceSize, VkDeviceSize> fuseBuffer, int bindIndex) {
    writeBuffer(std::get<0>(fuseBuffer), bindIndex, std::get<1>(fuseBuffer), std::get<2>(fuseBuffer));
}

void VulkanLayout::DescriptorSet::writeBuffer(VkBuffer buffer, int bindIndex, size_t size, VkDeviceSize offset) {
    VkWriteDescriptorSet writeSet;
    ::memset(&writeSet, 0, sizeof(writeSet));
    VkDescriptorBufferInfo sourceInfo;
    sourceInfo.buffer        = buffer;
    sourceInfo.offset        = offset;
    sourceInfo.range         = size;

    writeSet.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeSet.descriptorCount = 1;
    writeSet.descriptorType  = mPipeline->mBufferTypes[bindIndex];
    writeSet.dstBinding      = bindIndex;
    writeSet.pBufferInfo     = &sourceInfo;
    writeSet.dstSet          = mSet;

    mPipeline->mDevice.updateWriteDescriptorSet(writeSet);
}

void VulkanLayout::DescriptorSet::writeImage(VkImageView view, VkSampler sampler, VkImageLayout layout, int bind) {
    VkWriteDescriptorSet writeSet;
    ::memset(&writeSet, 0, sizeof(writeSet));
    VkDescriptorImageInfo sourceInfo;
    sourceInfo.imageView   = view;
    sourceInfo.imageLayout = layout;
    sourceInfo.sampler     = sampler;

    writeSet.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeSet.descriptorCount = 1;
    writeSet.descriptorType  = mPipeline->mBufferTypes[bind];
    writeSet.dstBinding      = bind;
    writeSet.pImageInfo      = &sourceInfo;
    writeSet.dstSet          = mSet;

    mPipeline->mDevice.updateWriteDescriptorSet(writeSet);
}

VulkanPipelineCache::VulkanPipelineCache(const VulkanDevice& dev) : mDevice(dev) {
    CALL_VK(mDevice.createPipelineCache(mCache));
}
VulkanPipelineCache::~VulkanPipelineCache() {
    mDevice.destroyPipelineCache(mCache);
}

VulkanLayout::~VulkanLayout() {
    mDevice.destroyPipelineLayout(mLayout);
    mDevice.destroyDescriptorSetLayout(mSetLayout);
}

VulkanLayout* VulkanLayout::create(const VulkanDevice& dev, const std::vector<VulkanLayout::LayoutType>& bufferTypesWithIndex) {
    // Turn bufferTypesWithIndex to bufferTypes;
    // Compute Max index for bufferTypesWithIndex
    int maxIndex = 0;
    for (auto& iter : bufferTypesWithIndex) {
        if (iter.binding > maxIndex) {
            maxIndex = iter.binding;
        }
    }
    std::vector<VkDescriptorType> bufferTypes(maxIndex+1, VK_DESCRIPTOR_TYPE_MAX_ENUM);
    for (auto& iter : bufferTypesWithIndex) {
        bufferTypes[iter.binding] = iter.type;
    }
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::map<VkDescriptorType, int> typeCount;
    for (auto& iter : bufferTypesWithIndex) {
        auto type = iter.type;
        if (typeCount.find(type) == typeCount.end()) {
            typeCount[type] = 1;
        } else {
            typeCount[type] += 1;
        }
        VkDescriptorSetLayoutBinding binding;
        binding.binding            = iter.binding;
        binding.descriptorType     = type;
        binding.descriptorCount    = 1;
        binding.stageFlags         = iter.stage;
        binding.pImmutableSamplers = nullptr;
        bindings.emplace_back(binding);
    }
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    {
        CALL_VK(dev.createDescriptorSetLayout(setLayout, bindings.size(), bindings.data()));
        CALL_VK(dev.createPipelineLayout(pipelineLayout, setLayout));
    }
    std::vector<VkDescriptorPoolSize> desPoolSize;
    for (auto& iter : typeCount) {
        VkDescriptorPoolSize s;
        s.descriptorCount = iter.second;
        s.type            = iter.first;
        desPoolSize.emplace_back(s);
    }
    auto layoutVk = new VulkanLayout(dev);
    layoutVk->mDesPoolSize = desPoolSize;
    layoutVk->mLayout = pipelineLayout;
    layoutVk->mSetLayout = setLayout;
    layoutVk->mBufferTypes = std::move(bufferTypes);
    return layoutVk;
}

VulkanGraphicPipelineCache* VulkanGraphicPipelineCache::create(const VulkanDevice& dev, const VulkanGraphicPipelineCache::ShaderSource& source) {
    auto cache = new VulkanGraphicPipelineCache(source.vertex, source.fragment, dev);
    return cache;
}

void VulkanGraphicPipelineCache::setVertexFormats(const std::vector<int>& units) {
    mVertexAttributes.resize(units.size());
    mVertexBindings.resize(units.size());
    for (int i=0; i<units.size(); ++i) {
        VkVertexInputAttributeDescription& attr = mVertexAttributes[i];
        auto unit = units[i];
        switch (unit) {
            case 4:
                attr.format = VK_FORMAT_R32G32B32A32_SFLOAT;
                break;
            case 3:
                attr.format = VK_FORMAT_R32G32B32_SFLOAT;
                break;
            case 2:
                attr.format = VK_FORMAT_R32G32_SFLOAT;
                break;
            case 1:
                attr.format = VK_FORMAT_R32_SFLOAT;
                break;
            default:
                break;
        }
        attr.binding = i;
        attr.offset = 0;
        attr.location = i;
        VkVertexInputBindingDescription& bind = mVertexBindings[i];
        bind.binding = i;
        bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bind.stride = unit * sizeof(float);
    }
    ::memset(&mVertexInfo, 0, sizeof(VkPipelineVertexInputStateCreateInfo));
    mVertexInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    mVertexInfo.pVertexAttributeDescriptions = mVertexAttributes.data();
    mVertexInfo.vertexAttributeDescriptionCount = (int)mVertexAttributes.size();
    mVertexInfo.vertexBindingDescriptionCount = (int)mVertexBindings.size();
    mVertexInfo.pVertexBindingDescriptions = mVertexBindings.data();
}

VulkanGraphicPipelineCache::VulkanGraphicPipelineCache(SharedPtr<VulkanShaderModule> vertex, SharedPtr<VulkanShaderModule> frag, const VulkanDevice& dev) : mDevice(dev) {
    mName = "main";
    mVertex = vertex;
    mFragment = frag;
    ::memset(mStage, 0, 2 * sizeof(VkPipelineShaderStageCreateInfo));
    mStage[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    mStage[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    mStage[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    mStage[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    mStage[0].module = vertex->get();
    mStage[1].module = frag->get();
    mStage[0].pName = mName.c_str();
    mStage[1].pName = mName.c_str();

    ::memset(&mBlend, 0, sizeof(VkPipelineColorBlendAttachmentState));
    // TODO: Set blend Info
    mBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    mBlend.logicOp = VK_LOGIC_OP_COPY;
    mBlend.logicOpEnable = VK_FALSE;
    mBlendAttchmentState.resize(4);
    mBlend.pAttachments = mBlendAttchmentState.data();
    ::memset(mBlendAttchmentState.data(), 0, mBlendAttchmentState.size() * sizeof(VkPipelineColorBlendAttachmentState));
    mBlend.attachmentCount = 1;
    for (int i=0; i<mBlendAttchmentState.size(); ++i) {
        mBlendAttchmentState[i].colorWriteMask = VK_COLOR_COMPONENT_A_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_R_BIT;
        mBlendAttchmentState[i].blendEnable = VK_FALSE;
    }

    ::memset(&mDepth, 0, sizeof(VkPipelineDepthStencilStateCreateInfo));
    mDepth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    mDepth.depthTestEnable = VK_TRUE;
    mDepth.depthWriteEnable = VK_TRUE;
    mDepth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    mDepth.stencilTestEnable = VK_FALSE;

    ::memset(&mRasterization, 0, sizeof(VkPipelineRasterizationStateCreateInfo));
    mRasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    mRasterization.polygonMode = VK_POLYGON_MODE_FILL;
    mRasterization.rasterizerDiscardEnable = VK_FALSE;
    mRasterization.depthClampEnable = VK_FALSE;
    mRasterization.depthBiasEnable = VK_FALSE;
    mRasterization.cullMode = VK_CULL_MODE_NONE;

    ::memset(&mInfo, 0, sizeof(VkGraphicsPipelineCreateInfo));
    mInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    mInfo.pStages = mStage;
    mInfo.stageCount = 2;
    mInfo.pVertexInputState = &mVertexInfo;
    mInfo.pDepthStencilState = &mDepth;
    mInfo.pColorBlendState = &mBlend;
    mInfo.pRasterizationState = &mRasterization;
    
    VkPipelineInputAssemblyStateCreateInfo& inputAssembly = mInputAssembly;
    ::memset(&inputAssembly, 0, sizeof(VkPipelineInputAssemblyStateCreateInfo));
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    mInfo.pInputAssemblyState = &inputAssembly;
}

VulkanGraphicPipelineCache::~VulkanGraphicPipelineCache() {
    // Do nothing
}

VulkanShaderModule::VulkanShaderModule(VkShaderModule shader, const VulkanDevice& dev) : mShader(shader), mDevice(dev) {
    // Do nothing
}
VulkanShaderModule::~VulkanShaderModule() {
    mDevice.destroyShaderModule(mShader);
}
VulkanShaderModule* VulkanShaderModule::create(const VulkanDevice& dev, const uint32_t* buffer, size_t size) {
    VkShaderModule shaderOut;
    VkResult result;
    result = dev.createShaderModule(shaderOut, size, (const uint32_t*)buffer);
    if (VK_SUCCESS != result) {
        MNN_ERROR("Create Vulkan Shader error: %d\n", result);
        return nullptr;
    }
    return new VulkanShaderModule(shaderOut, dev);
}


} // namespace MNN
