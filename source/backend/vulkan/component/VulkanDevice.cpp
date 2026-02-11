//
//  VulkanDevice.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/component/VulkanDevice.hpp"
#include <string.h>
#include <algorithm>
//#define MNN_VULKAN_PRINT_EXT
namespace MNN {
static uint32_t _getLocalMemorySize(const VkPhysicalDeviceMemoryProperties& memProty) {
#ifdef __APPLE__
    // For mac vulkan driver can not get correct local size
    return 16384;
#else
    int32_t localMemorySize = 0;
    for (int i=0; i<memProty.memoryHeapCount; ++i) {
        auto& heap = memProty.memoryHeaps[i];
        if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            auto size = (int32_t)heap.size;
            if (size > 0) {
                localMemorySize = size;
                break;
            }
        }
    }
    return localMemorySize;
#endif
}

static bool _hasExtension(const std::vector<VkExtensionProperties>& exts, const char* name) {
    return std::any_of(exts.begin(), exts.end(), [&](const VkExtensionProperties& ext) {
        return std::strcmp(ext.extensionName, name) == 0;
    });
}

VulkanDevice::VulkanDevice(std::shared_ptr<VulkanInstance> instance)
    : mOwner(true),
      mInstance(instance),
      mQueueFamilyIndex(0),
      mPhysicalDevice(VK_NULL_HANDLE),
      mDevice(VK_NULL_HANDLE),
      mQueue(VK_NULL_HANDLE) {
    // Find one GPU to use:
    // On Android, every GPU device is equal -- supporting
    // graphics/compute/present
    // for this sample, we use the very first GPU device found on the system
    uint32_t gpuCount = 0;
    CALL_VK(mInstance->enumeratePhysicalDevices(gpuCount, nullptr));
    MNN_ASSERT(0 != gpuCount);
    std::vector<VkPhysicalDevice> tmpGpus(gpuCount);
    CALL_VK(mInstance->enumeratePhysicalDevices(gpuCount, tmpGpus.data()));
    MNN_ASSERT(nullptr != tmpGpus[0]);
    mPhysicalDevice = tmpGpus[0];

    // Set queue.
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
        if (!(queueFamilyProperties[queueFamilyIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            MNN_PRINT("The queue can't support graphic render\n");
        }
    }
    MNN_ASSERT(queueFamilyIndex < queueFamilyCount);
    mQueueFamilyIndex = queueFamilyIndex;
    float priorities[] = {
        1.0f,
    };
    VkDeviceQueueCreateInfo queueCreateInfo{
        /* .sType            = */ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        /* .pNext            = */ nullptr,
        /* .flags            = */ 0,
        /* .queueFamilyIndex = */ mQueueFamilyIndex,
        /* .queueCount       = */ 1,
        /* .pQueuePriorities = */ priorities,
    };

    // Set device features.
    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    
    VkPhysicalDeviceFeatures2 deviceFeatures2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    deviceFeatures2.features = deviceFeatures;

    void* pNextChain = nullptr;

    // Set device extensions.
    std::vector<const char*> deviceExtensions;
    std::vector<VkExtensionProperties> availableDeviceExtensions;
    {
        uint32_t extCount = 0;
        CALL_VK(vkEnumerateDeviceExtensionProperties(mPhysicalDevice, nullptr, &extCount, nullptr));
        availableDeviceExtensions.resize(extCount);
        CALL_VK(vkEnumerateDeviceExtensionProperties(mPhysicalDevice, nullptr, &extCount, availableDeviceExtensions.data()));
    }

    // Configure VK_KHR_portability_subset
    const char * portabilityExtName = "VK_KHR_portability_subset";
    if (_hasExtension(availableDeviceExtensions, portabilityExtName)) {
        deviceExtensions.push_back(portabilityExtName);
    }

    // Configure FP16
    checkFP16(availableDeviceExtensions);
    if (mFP16Info.supportFP16) {
        if (mFP16Info.FP16FromExtension) {
            deviceExtensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
            deviceExtensions.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);

            // Chain KHR structs
            mFP16Info.enabledShaderFloat16Int8Features.pNext = pNextChain;
            pNextChain = &mFP16Info.enabledShaderFloat16Int8Features;
            
            mFP16Info.enabled16BitStorageFeatures.pNext = pNextChain;
            pNextChain = &mFP16Info.enabled16BitStorageFeatures;
        } else {
            // Chain Core structs
            mFP16Info.enabledVulkan12Features.pNext = pNextChain;
            pNextChain = &mFP16Info.enabledVulkan12Features;

            mFP16Info.enabledVulkan11Features.pNext = pNextChain;
            pNextChain = &mFP16Info.enabledVulkan11Features;
        }
    }

    // Configure coopMat
    checkCoopMat(availableDeviceExtensions);
    if (mCoopMatInfo.supportCoopMat) {
        deviceExtensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        mCoopMatInfo.enabledCoopMatFeatures.pNext = pNextChain;
        pNextChain = &mCoopMatInfo.enabledCoopMatFeatures;
    }

    deviceFeatures2.pNext = pNextChain;

    // Create Device. Get Queue.
    VkDeviceCreateInfo deviceCreateInfo{
        /* .sType                   = */ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        /* .pNext                   = */ nullptr,
        /* .flags                   = */ 0,
        /* .queueCreateInfoCount    = */ 1,
        /* .pQueueCreateInfos       = */ &queueCreateInfo,
        /* .enabledLayerCount       = */ 0,
        /* .ppEnabledLayerNames     = */ nullptr,
        /* .enabledExtensionCount   = */ static_cast<uint32_t>(deviceExtensions.size()),
        /* .ppEnabledExtensionNames = */ deviceExtensions.data(),
        /* .pEnabledFeatures        = */ nullptr,
    };
    deviceCreateInfo.pNext = &deviceFeatures2;
    mDevice = VK_NULL_HANDLE;
    CALL_VK(vkCreateDevice(mPhysicalDevice, &deviceCreateInfo, nullptr, &mDevice));
    if (VK_NULL_HANDLE == mDevice) {
        MNN_ERROR("Can't create vk device\n");
        return;
    }
    getDeviceQueue(mQueueFamilyIndex, 0, mQueue);

    // Query device properties.
    vkGetPhysicalDeviceProperties(mPhysicalDevice, &mDeviceProty);
    vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &mMemoryProty);
    mLocalMemorySize = _getLocalMemorySize(mMemoryProty);

    // query subgroupSize
    {
        VkPhysicalDeviceProperties2 deviceProperties2 = {};
        deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

        VkPhysicalDeviceSubgroupProperties subgroupProperties = {};
        subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

        deviceProperties2.pNext = &subgroupProperties;
        vkGetPhysicalDeviceProperties2(mPhysicalDevice, &deviceProperties2);
        mSubgroupSize = subgroupProperties.subgroupSize;
    }
#ifdef MNN_VULKAN_PRINT_EXT
    uint32_t pPropertyCount;
    vkEnumerateInstanceExtensionProperties(nullptr, &pPropertyCount, nullptr);
    std::vector<VkExtensionProperties> properties(pPropertyCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &pPropertyCount, properties.data());
    for (int i=0; i<pPropertyCount; ++i) {
      auto& p = properties[i];
      FUNC_PRINT_ALL(p.extensionName, s);
    }
    FUNC_PRINT(mDeviceProty.limits.maxComputeWorkGroupSize[0]);
    FUNC_PRINT(mDeviceProty.limits.maxComputeWorkGroupCount[0]);
    FUNC_PRINT(mDeviceProty.limits.maxComputeWorkGroupInvocations);
    FUNC_PRINT(mDeviceProty.limits.maxComputeSharedMemorySize);
    FUNC_PRINT(mLocalMemorySize);
#endif
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
      vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &mMemoryProty);
      mLocalMemorySize = _getLocalMemorySize(mMemoryProty);
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

const VkResult VulkanDevice::createCommandPool(VkCommandPool& cmdPool, const VkCommandPoolCreateFlags flags,
                                               const VkAllocationCallbacks* allocator) const {
    VkCommandPoolCreateInfo cmdPoolCreateInfo{
        /* .sType            = */ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        /* .pNext            = */ nullptr,
        /* .flags            = */ flags,
        /* .queueFamilyIndex = */ mQueueFamilyIndex,
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
        /* .sType              = */ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        /* .pNext              = */ nullptr,
        /* .commandPool        = */ cmdPool,
        /* .level              = */ level,
        /* .commandBufferCount = */ cmdBufferCount,
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
        /* .sType = */ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        /* .pNext = */ nullptr,
        /* .flags = */ 0,
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
                                         const uint32_t height, const uint32_t depth, const VkFormat format, VkImageUsageFlags usage,
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
    info.usage             = usage;
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
        /* .sType           = */ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        /* .pNext           = */ nullptr,
        /* .flags           = */ 0, // reserved, must be 0
        /* .initialDataSize = */ 0,
        /* .pInitialData    = */ nullptr,
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
        /* .sType    = */ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        /* .pNext    = */ nullptr,
        /* .flags    = */ 0,
        /* .codeSize = */ codeSize,
        /* .pCode    = */ pCode,
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
    poolInfo.flags                      = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    return vkCreateDescriptorPool(mDevice, &poolInfo, allocator, &descriptorPool);
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
    return vkAllocateDescriptorSets(mDevice, &allocInfo, &descriptorSet);
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

void VulkanDevice::checkFP16(const std::vector<VkExtensionProperties>& availableExts) {
    mFP16Info.supportFP16 = false;
    mFP16Info.FP16FromExtension = false;
    mFP16Info.enabledVulkan11Features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    mFP16Info.enabledVulkan12Features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    mFP16Info.enabledShaderFloat16Int8Features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    mFP16Info.enabled16BitStorageFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    
    PFN_vkGetPhysicalDeviceFeatures2 getFeatures2 = vkGetPhysicalDeviceFeatures2;
    if (!getFeatures2) {
        getFeatures2 = vkGetPhysicalDeviceFeatures2KHR;
    }
    if (!getFeatures2) {
        return;
    }

    // 1. Try Vulkan 1.2 Core approach
    {
        VkPhysicalDeviceVulkan11Features vk11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceVulkan12Features vk12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        vk12.pNext = &vk11;

        VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features2.pNext = &vk12;
        getFeatures2(mPhysicalDevice, &features2);

        if (vk12.shaderFloat16 == VK_TRUE && vk11.storageBuffer16BitAccess == VK_TRUE) {
            mFP16Info.supportFP16 = true;
            mFP16Info.enabledVulkan12Features.shaderFloat16 = VK_TRUE;
            mFP16Info.enabledVulkan11Features.storageBuffer16BitAccess = VK_TRUE;
            return;
        }
    }

    // 2. Try KHR Extension approach
    {
        if (!_hasExtension(availableExts, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) ||
            !_hasExtension(availableExts, VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) {
            return;
        }

        VkPhysicalDeviceShaderFloat16Int8Features khrFloat16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDevice16BitStorageFeatures khrStorage = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
        khrFloat16.pNext = &khrStorage;

        VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features2.pNext = &khrFloat16;
        getFeatures2(mPhysicalDevice, &features2);

        if (khrFloat16.shaderFloat16 == VK_TRUE && khrStorage.storageBuffer16BitAccess == VK_TRUE) {
            mFP16Info.supportFP16 = true;
            mFP16Info.FP16FromExtension = true;
            mFP16Info.enabledShaderFloat16Int8Features.shaderFloat16 = VK_TRUE;
            mFP16Info.enabled16BitStorageFeatures.storageBuffer16BitAccess = VK_TRUE;
            return;
        }
    }
}

void VulkanDevice::checkCoopMat(const std::vector<VkExtensionProperties>& availableExts) {
    mCoopMatInfo.supportCoopMat = false;
    mCoopMatInfo.enabledCoopMatFeatures = {};
    mCoopMatInfo.enabledCoopMatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    mCoopMatInfo.fp32CoopMatShape.clear();
    mCoopMatInfo.fp16CoopMatShape.clear();
    mCoopMatInfo.selectedFP32CoopMatShape.clear();
    mCoopMatInfo.selectedFP16CoopMatShape.clear();

    PFN_vkGetPhysicalDeviceFeatures2 getFeatures2 = vkGetPhysicalDeviceFeatures2;
    if (!getFeatures2) {
        getFeatures2 = vkGetPhysicalDeviceFeatures2KHR;
    }
    if (!getFeatures2) {
        return;
    }

    if (!_hasExtension(availableExts, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
        return;
    }

    // 2. Check Feature
    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &mCoopMatInfo.enabledCoopMatFeatures;
    
    getFeatures2(mPhysicalDevice, &features2);

    if (mCoopMatInfo.enabledCoopMatFeatures.cooperativeMatrix != VK_TRUE) return;

    // 3. Query Properties (Shapes)
    VkInstance instance = mInstance->get();
    auto fpGetCoopMat = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));

    if (!fpGetCoopMat) return;

    uint32_t propCount = 0;
    if (fpGetCoopMat(mPhysicalDevice, &propCount, nullptr) != VK_SUCCESS || propCount == 0) return;

    std::vector<VkCooperativeMatrixPropertiesKHR> props(propCount);
    for (auto& p : props) {
        p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        p.pNext = nullptr;
    }
    fpGetCoopMat(mPhysicalDevice, &propCount, props.data());

    uint32_t maxFP16Size = 0;
    uint32_t maxFP32Size = 0;

    for (const auto & p : props) {
        if (p.scope != VK_SCOPE_SUBGROUP_KHR || p.saturatingAccumulation != VK_FALSE) continue;

        bool isFP16 = (p.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && p.BType == VK_COMPONENT_TYPE_FLOAT16_KHR && p.CType == VK_COMPONENT_TYPE_FLOAT16_KHR && p.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR);
        bool isFP32 = (p.AType == VK_COMPONENT_TYPE_FLOAT32_KHR && p.BType == VK_COMPONENT_TYPE_FLOAT32_KHR && p.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && p.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR);

        uint32_t size = p.MSize * p.NSize * p.KSize;

        if (isFP16) {
            mCoopMatInfo.fp16CoopMatShape.push_back({p.MSize, p.NSize, p.KSize});
            if (size > maxFP16Size) {
                maxFP16Size = size;
                mCoopMatInfo.selectedFP16CoopMatShape = {p.MSize, p.NSize, p.KSize};
            }
        }
        if (isFP32) {
            mCoopMatInfo.fp32CoopMatShape.push_back({p.MSize, p.NSize, p.KSize});
            if (size > maxFP32Size) {
                maxFP32Size = size;
                mCoopMatInfo.selectedFP32CoopMatShape = {p.MSize, p.NSize, p.KSize};
            }
        }
    }

    mCoopMatInfo.supportCoopMat = true;
}

} // namespace MNN
