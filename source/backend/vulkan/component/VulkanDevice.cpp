//
//  VulkanDevice.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/component/VulkanDevice.hpp"
#include <string.h>

namespace MNN {
VulkanDevice::VulkanDevice(std::shared_ptr<VulkanInstance> instance, const std::vector<const char*>& device_extensions)
    : mOwner(true),
      mInstance(instance),
      mQueueFamilyIndex(0),
      mPhysicalDevice(VK_NULL_HANDLE),
      mDevice(VK_NULL_HANDLE),
      mQueue(VK_NULL_HANDLE) {
    MNN_ASSERT(mInstance->success());
    // Find one GPU to use:
    // On Android, every GPU device is equal -- supporting
    // graphics/compute/present
    // for this sample, we use the very first GPU device found on the system
    uint32_t gpuCount = 0;
    CALL_VK(mInstance->enumeratePhysicalDevices(gpuCount, nullptr));
    MNN_ASSERT(0 != gpuCount);
    VkPhysicalDevice tmpGpus[gpuCount];
    CALL_VK(mInstance->enumeratePhysicalDevices(gpuCount, tmpGpus));
    MNN_ASSERT(0 != gpuCount);
    mPhysicalDevice = tmpGpus[0];

    // Find a GFX queue family
    uint32_t queueFamilyCount = 1;
    uint32_t queueFamilyIndex = 0;

    mInstance->getPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, queueFamilyCount, nullptr);

    MNN_ASSERT(queueFamilyCount);
    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
    mInstance->getPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, queueFamilyCount, queueFamilyProperties.data());

    for (queueFamilyIndex = 0; queueFamilyIndex < queueFamilyCount; queueFamilyIndex++) {
        if (queueFamilyProperties[queueFamilyIndex].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            break;
        }
    }
    MNN_ASSERT(queueFamilyIndex < queueFamilyCount);
    mQueueFamilyIndex = queueFamilyIndex;

    // Create a logical device (vulkan device)
    float priorities[] = {
        1.0f,
    };
    VkDeviceQueueCreateInfo queueCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = 0,
        .queueFamilyIndex = mQueueFamilyIndex,
        .queueCount       = 1,
        .pQueuePriorities = priorities,
    };

    VkDeviceCreateInfo deviceCreateInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = nullptr,
        .flags                   = 0,
        .queueCreateInfoCount    = 1,
        .pQueueCreateInfos       = &queueCreateInfo,
        .enabledLayerCount       = 0,
        .ppEnabledLayerNames     = nullptr,
        .enabledExtensionCount   = static_cast<uint32_t>(device_extensions.size()),
        .ppEnabledExtensionNames = device_extensions.data(),
        .pEnabledFeatures        = nullptr,
    };

    CALL_VK(vkCreateDevice(mPhysicalDevice, &deviceCreateInfo, nullptr, &mDevice));
    vkGetPhysicalDeviceProperties(mPhysicalDevice, &mDeviceProty);
    getDeviceQueue(mQueueFamilyIndex, 0, mQueue);
}

VulkanDevice::VulkanDevice(std::shared_ptr<VulkanInstance> instance, VkPhysicalDevice physicalDevice, VkDevice device,
                           uint32_t queueFamilyIndex, VkQueue queue)
    : mOwner(false),
      mInstance(instance),
      mQueueFamilyIndex(queueFamilyIndex),
      mPhysicalDevice(physicalDevice),
      mDevice(device),
      mQueue(queue) {
    vkGetPhysicalDeviceProperties(mPhysicalDevice, &mDeviceProty);
}

VulkanDevice::~VulkanDevice() {
    if (mOwner && (VK_NULL_HANDLE != mDevice)) {
        vkDestroyDevice(mDevice, nullptr);
        mDevice = VK_NULL_HANDLE;
    }
}

void VulkanDevice::getDeviceQueue(const uint32_t familyIndex, const uint32_t queueIndex, VkQueue& queue) {
    vkGetDeviceQueue(get(), familyIndex, queueIndex, &queue);
}

const VkQueue VulkanDevice::acquireDefaultDevQueue() const {
    return mQueue;
}

const VkResult VulkanDevice::createBuffer(VkBuffer& buffer, const size_t size, const VkBufferUsageFlags usage,
                                          const VkSharingMode shared, const VkAllocationCallbacks* allocator) const {
    VkBufferCreateInfo info    = {};
    info.sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.pNext                 = nullptr;
    info.flags                 = 0;
    info.size                  = (VkDeviceSize)size;
    info.usage                 = usage;
    info.sharingMode           = shared;
    info.pQueueFamilyIndices   = &mQueueFamilyIndex;
    info.queueFamilyIndexCount = 1;
    return vkCreateBuffer(mDevice, &info, allocator, &buffer);
}

const void VulkanDevice::getBufferMemoryRequirements(VkBuffer buffer, VkMemoryRequirements& memoryRequirements) const {
    vkGetBufferMemoryRequirements(mDevice, buffer, &memoryRequirements);
}

const VkResult VulkanDevice::allocMemory(VkDeviceMemory& memory, const VkMemoryAllocateInfo& allocateInfo,
                                         const VkAllocationCallbacks* allocator) const {
    return vkAllocateMemory(mDevice, &allocateInfo, allocator, &memory);
}

const void VulkanDevice::freeMemory(const VkDeviceMemory& memory, const VkAllocationCallbacks* allocator) const {
    vkFreeMemory(mDevice, memory, allocator);
}

const VkResult VulkanDevice::mapMemory(const VkDeviceMemory memory, const VkDeviceSize offset, const VkDeviceSize size,
                                       const VkMemoryMapFlags flags, void** ppData) const {
    return vkMapMemory(mDevice, memory, offset, size, flags, ppData);
}

const void VulkanDevice::unmapMemory(const VkDeviceMemory memory) const {
    vkUnmapMemory(mDevice, memory);
}

const VkResult VulkanDevice::bindBufferMemory(const VkBuffer buffer, const VkDeviceMemory memory,
                                              const VkDeviceSize memoryOffset) const {
    return vkBindBufferMemory(mDevice, buffer, memory, memoryOffset);
}

const void VulkanDevice::destroyBuffer(const VkBuffer buffer, const VkAllocationCallbacks* allocator) const {
    vkDestroyBuffer(mDevice, buffer, allocator);
}

const VkResult VulkanDevice::flushMappedMemoryRanges(const VkMappedMemoryRange* memoryRanges,
                                                     const uint32_t memoryRangeCount) const {
    return vkFlushMappedMemoryRanges(mDevice, memoryRangeCount, memoryRanges);
}

const VkResult VulkanDevice::invalidateMappedMemoryRanges(const VkMappedMemoryRange* memoryRanges,
                                                          const uint32_t memoryRangeCount) const {
    return vkInvalidateMappedMemoryRanges(mDevice, memoryRangeCount, memoryRanges);
}

const void VulkanDevice::getPhysicalDeviceMemoryProperties(VkPhysicalDeviceMemoryProperties& memoryProperties) const {
    vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &memoryProperties);
}

const VkResult VulkanDevice::createCommandPool(VkCommandPool& cmdPool, const VkCommandPoolCreateFlags flags,
                                               const VkAllocationCallbacks* allocator) const {
    VkCommandPoolCreateInfo cmdPoolCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = flags,
        .queueFamilyIndex = mQueueFamilyIndex,
    };
    return vkCreateCommandPool(mDevice, &cmdPoolCreateInfo, allocator, &cmdPool);
}

const void VulkanDevice::destroyCommandPool(const VkCommandPool& cmdPool,
                                            const VkAllocationCallbacks* allocator) const {
    vkDestroyCommandPool(mDevice, cmdPool, allocator);
}

const VkResult VulkanDevice::allocateCommandBuffers(const VkCommandPool& cmdPool, VkCommandBuffer* cmdBuffers,
                                                    const uint32_t cmdBufferCount,
                                                    const VkCommandBufferLevel level) const {
    VkCommandBufferAllocateInfo cmdBufferCreateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = cmdPool,
        .level              = level,
        .commandBufferCount = cmdBufferCount,
    };
    return vkAllocateCommandBuffers(mDevice, &cmdBufferCreateInfo, cmdBuffers);
}

const void VulkanDevice::freeCommandBuffers(const VkCommandPool& cmdPool, const VkCommandBuffer* cmdBuffers,
                                            const uint32_t cmdBufferCount) const {
    vkFreeCommandBuffers(mDevice, cmdPool, cmdBufferCount, cmdBuffers);
}

const VkResult VulkanDevice::allocateCommandBuffer(const VkCommandPool& cmdPool, VkCommandBuffer& cmdBuffer,
                                                   const VkCommandBufferLevel level) const {
    return allocateCommandBuffers(cmdPool, &cmdBuffer, 1, level);
}
const void VulkanDevice::freeCommandBuffer(const VkCommandPool& cmdPool, const VkCommandBuffer& cmdBuffer) const {
    freeCommandBuffers(cmdPool, &cmdBuffer, 1);
}

const VkResult VulkanDevice::createFence(VkFence& fence, const VkAllocationCallbacks* allocator) const {
#ifdef VK_USE_PLATFORM_WIN32_KHR
    // which one is correct on windows ?
    VkExportFenceCreateInfoKHR efci;
    // VkExportFenceWin32HandleInfoKHR efci;
    efci.sType = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO;
    efci.pNext = NULL;
    efci.sType = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
#else
    VkExportFenceCreateInfoKHR efci;
    efci.sType       = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO;
    efci.pNext       = NULL;
#if VK_USE_PLATFORM_ANDROID_KHR // current android only support VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT
    efci.handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
#else
    efci.handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
#endif
    VkFenceCreateInfo fci{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };
    return vkCreateFence(mDevice, &fci, allocator, &fence);
}

const VkResult VulkanDevice::waitForFence(const VkFence& fence, const uint64_t timeout) const {
    return waitForFences(1, &fence, VK_TRUE, timeout);
}
const VkResult VulkanDevice::waitForFences(const uint32_t fenceCount, const VkFence* fences, const VkBool32 waitAll,
                                           const uint64_t timeout) const {
    return vkWaitForFences(mDevice, fenceCount, fences, waitAll, timeout);
}

void VulkanDevice::destroyFence(const VkFence& fence, const VkAllocationCallbacks* allocator) const {
    vkDestroyFence(mDevice, fence, allocator);
}

const VkResult VulkanDevice::resetFences(const uint32_t fenceCount, const VkFence* fences) const {
    return vkResetFences(mDevice, fenceCount, fences);
}
const VkResult VulkanDevice::resetFence(const VkFence& fence) const {
    return resetFences(1, &fence);
}
const VkResult VulkanDevice::enumerateDeviceExtensionProperties(const VkPhysicalDevice& dev,
                                                                std::vector<VkExtensionProperties>& exts_props) const {
    uint32_t propertyCount = 0;
    VkResult result        = VK_SUCCESS;

    do {
        result = vkEnumerateDeviceExtensionProperties(dev, nullptr, &propertyCount, nullptr);
        if ((VK_SUCCESS == result) && propertyCount) {
            std::vector<VkExtensionProperties> props(propertyCount);
            result = vkEnumerateDeviceExtensionProperties(dev, nullptr, &propertyCount,
                                                          reinterpret_cast<VkExtensionProperties*>(props.data()));
            if ((VK_SUCCESS == result) && propertyCount) {
                exts_props.insert(exts_props.end(), props.begin(), props.end());
            }
        }
    } while (VK_INCOMPLETE == result);

    return result;
}

const VkResult VulkanDevice::createSemaphore(VkSemaphore& semaphore, const VkAllocationCallbacks* allocator) const {
    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreInfo.flags                 = 0;
    semaphoreInfo.pNext                 = nullptr;
    return vkCreateSemaphore(mDevice, &semaphoreInfo, allocator, &semaphore);
}

const void VulkanDevice::destroySemaphore(const VkSemaphore& semaphore, const VkAllocationCallbacks* allocator) const {
    vkDestroySemaphore(mDevice, semaphore, allocator);
}

const VkResult VulkanDevice::createImage(VkImage& image, const VkImageType imageType, const uint32_t width,
                                         const uint32_t height, const uint32_t depth, const VkFormat format,
                                         const VkAllocationCallbacks* allocator) const {
    VkImageCreateInfo info = {};
    info.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType         = imageType;
    info.extent.width      = width;
    info.extent.height     = height;
    info.extent.depth      = depth;
    info.mipLevels         = 1;
    info.arrayLayers       = 1;
    info.format            = format;
    info.tiling            = VK_IMAGE_TILING_OPTIMAL;
    info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    info.usage             = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.samples           = VK_SAMPLE_COUNT_1_BIT;
    info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
    info.pNext             = nullptr;
    return vkCreateImage(mDevice, &info, allocator, &image);
}

const void VulkanDevice::destroyImage(const VkImage& image, const VkAllocationCallbacks* allocator) const {
    vkDestroyImage(mDevice, image, allocator);
}

const void VulkanDevice::getImageMemoryRequirements(const VkImage& image,
                                                    VkMemoryRequirements& memoryRequirements) const {
    vkGetImageMemoryRequirements(mDevice, image, &memoryRequirements);
}

const void VulkanDevice::bindImageMemory(const VkImage& image, const VkDeviceMemory& memory,
                                         const VkDeviceSize& memoryOffset) const {
    vkBindImageMemory(mDevice, image, memory, memoryOffset);
}

const VkResult VulkanDevice::createImageView(VkImageView& view, const VkImage& image, const VkImageViewType& viewType,
                                             const VkFormat& format, const VkAllocationCallbacks* allocator) const {
    VkImageViewCreateInfo info           = {};
    info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image                           = image;
    info.viewType                        = viewType;
    info.format                          = format;
    info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    info.subresourceRange.baseMipLevel   = 0;
    info.subresourceRange.levelCount     = 1;
    info.subresourceRange.baseArrayLayer = 0;
    info.subresourceRange.layerCount     = 1;

    return vkCreateImageView(mDevice, &info, allocator, &view);
}

const void VulkanDevice::destroyImageView(const VkImageView& imageView, const VkAllocationCallbacks* allocator) const {
    vkDestroyImageView(mDevice, imageView, allocator);
}

const VkResult VulkanDevice::createSampler(VkSampler& sampler, const VkFilter& filter, const VkSamplerAddressMode& mode,
                                           const VkAllocationCallbacks* allocator) const {
    VkSamplerCreateInfo samplerInfo;
    ::memset(&samplerInfo, 0, sizeof(samplerInfo));
    samplerInfo.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter        = filter;
    samplerInfo.minFilter        = filter;
    samplerInfo.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU     = mode;
    samplerInfo.addressModeV     = mode;
    samplerInfo.addressModeW     = mode;
    samplerInfo.mipLodBias       = 0.0f;
    samplerInfo.borderColor      = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy    = 1.0f;
    samplerInfo.compareEnable    = VK_FALSE;
    samplerInfo.minLod           = 0.0f;
    samplerInfo.maxLod           = 0.0f;
    return vkCreateSampler(mDevice, &samplerInfo, allocator, &sampler);
}

const void VulkanDevice::destroySampler(const VkSampler& sampler, const VkAllocationCallbacks* allocator) const {
    vkDestroySampler(mDevice, sampler, allocator);
}

const VkResult VulkanDevice::createPipelineCache(VkPipelineCache& pipelineCache,
                                                 const VkAllocationCallbacks* allocator) const {
    VkPipelineCacheCreateInfo pipelineCacheInfo{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .pNext           = nullptr,
        .flags           = 0, // reserved, must be 0
        .initialDataSize = 0,
        .pInitialData    = nullptr,
    };
    return vkCreatePipelineCache(mDevice, &pipelineCacheInfo, allocator, &pipelineCache);
}

const void VulkanDevice::destroyPipelineCache(const VkPipelineCache& pipelineCache,
                                              const VkAllocationCallbacks* allocator) const {
    vkDestroyPipelineCache(mDevice, pipelineCache, allocator);
}

const VkResult VulkanDevice::createShaderModule(VkShaderModule& shaderModule, const size_t codeSize,
                                                const uint32_t* pCode, const VkAllocationCallbacks* allocator) const {
    VkShaderModuleCreateInfo shaderModuleCreateInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .codeSize = codeSize,
        .pCode    = pCode,
    };
    return vkCreateShaderModule(mDevice, &shaderModuleCreateInfo, allocator, &shaderModule);
}

const void VulkanDevice::destroyShaderModule(const VkShaderModule& shaderModule,
                                             const VkAllocationCallbacks* allocator) const {
    vkDestroyShaderModule(mDevice, shaderModule, allocator);
}

const void VulkanDevice::updateDescriptorSets(uint32_t descriptorWriteCount,
                                              const VkWriteDescriptorSet* pDescriptorWrites,
                                              uint32_t descriptorCopyCount,
                                              const VkCopyDescriptorSet* pDescriptorCopies) const {
    vkUpdateDescriptorSets(mDevice, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);
}

const void VulkanDevice::updateWriteDescriptorSet(const VkWriteDescriptorSet& descriptorWrite) const {
    updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
}

const VkResult VulkanDevice::createDescriptorSetLayout(VkDescriptorSetLayout& setLayout, const uint32_t bindingCount,
                                                       const VkDescriptorSetLayoutBinding* bindings,
                                                       const VkAllocationCallbacks* allocator) const {
    VkDescriptorSetLayoutCreateInfo info;
    info.bindingCount = bindingCount;
    info.pBindings    = bindings;
    info.pNext        = nullptr;
    info.flags        = 0;
    info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;

    return vkCreateDescriptorSetLayout(mDevice, &info, allocator, &setLayout);
}

const VkResult VulkanDevice::createPipelineLayout(VkPipelineLayout& pipelineLayout,
                                                  const VkDescriptorSetLayout& setLayout,
                                                  const VkAllocationCallbacks* allocator) const {
    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount             = 1;
    layoutInfo.pSetLayouts                = &setLayout;
    return vkCreatePipelineLayout(mDevice, &layoutInfo, allocator, &pipelineLayout);
}

const void VulkanDevice::destroyPipelineLayout(const VkPipelineLayout& pipelineLayout,
                                               const VkAllocationCallbacks* allocator) const {
    vkDestroyPipelineLayout(mDevice, pipelineLayout, allocator);
}

const VkResult VulkanDevice::createComputePipelines(VkPipeline* pipelines,
                                                    const VkComputePipelineCreateInfo* createInfos,
                                                    const uint32_t createInfoCount,
                                                    const VkPipelineCache& pipelineCache,
                                                    const VkAllocationCallbacks* allocator) const {
    return vkCreateComputePipelines(mDevice, pipelineCache, createInfoCount, createInfos, allocator, pipelines);
}

const VkResult VulkanDevice::createComputePipeline(VkPipeline& pipeline, const VkShaderModule& shaderMoule,
                                                   const VkPipelineLayout& pipelineLayout,
                                                   const VkPipelineCache& pipelineCache,
                                                   const VkSpecializationInfo* pSpecializationInfo,
                                                   const VkAllocationCallbacks* allocator) const {
    VkComputePipelineCreateInfo info;
    ::memset(&info, 0, sizeof(info));
    info.sType                     = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage               = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module              = shaderMoule;
    info.stage.pName               = "main";
    info.layout                    = pipelineLayout;
    info.stage.pSpecializationInfo = pSpecializationInfo;

    return createComputePipelines(&pipeline, &info, 1, pipelineCache, allocator);
}

const void VulkanDevice::destroyDescriptorSetLayout(const VkDescriptorSetLayout& descriptorSetLayout,
                                                    const VkAllocationCallbacks* allocator) const {
    vkDestroyDescriptorSetLayout(mDevice, descriptorSetLayout, allocator);
}
const void VulkanDevice::destroyPipeline(const VkPipeline& pipeline, const VkAllocationCallbacks* allocator) const {
    vkDestroyPipeline(mDevice, pipeline, allocator);
}

const VkResult VulkanDevice::createDescriptorPool(VkDescriptorPool& descriptorPool, const uint32_t poolSizeCount,
                                                  const VkDescriptorPoolSize* pPoolSizes,
                                                  const VkAllocationCallbacks* allocator) const {
    VkDescriptorPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.poolSizeCount              = poolSizeCount;
    poolInfo.pPoolSizes                 = pPoolSizes;
    poolInfo.maxSets                    = 1;
    return vkCreateDescriptorPool(mDevice, &poolInfo, allocator, &descriptorPool);
}

const VkResult VulkanDevice::allocateDescriptorSets(VkDescriptorSet* pDescriptorSets,
                                                    const VkDescriptorSetAllocateInfo* allocateInfo) const {
    return vkAllocateDescriptorSets(mDevice, allocateInfo, pDescriptorSets);
}

const VkResult VulkanDevice::allocateDescriptorSet(VkDescriptorSet& descriptorSet,
                                                   const VkDescriptorSetAllocateInfo& allocateInfo) const {
    return allocateDescriptorSets(&descriptorSet, &allocateInfo);
}

const VkResult VulkanDevice::allocateDescriptorSet(VkDescriptorSet& descriptorSet, const VkDescriptorPool& descPool,
                                                   const VkDescriptorSetLayout& setLayout) const {
    VkDescriptorSetAllocateInfo allocInfo;
    ::memset(&allocInfo, 0, sizeof(allocInfo));
    allocInfo.pNext              = nullptr;
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = descPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &setLayout;
    ;
    return allocateDescriptorSet(descriptorSet, allocInfo);
}

const VkResult VulkanDevice::freeDescriptorSets(const VkDescriptorPool& descriptorPool,
                                                const uint32_t descriptorSetCount,
                                                const VkDescriptorSet* pDescriptorSets) const {
    return vkFreeDescriptorSets(mDevice, descriptorPool, descriptorSetCount, pDescriptorSets);
}

const void VulkanDevice::destroyDescriptorPool(const VkDescriptorPool& descriptorPool,
                                               const VkAllocationCallbacks* allocator) const {
    vkDestroyDescriptorPool(mDevice, descriptorPool, allocator);
}

} // namespace MNN
