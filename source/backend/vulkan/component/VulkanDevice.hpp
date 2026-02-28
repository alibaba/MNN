//
//  VulkanDevice.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanDevice_hpp
#define VulkanDevice_hpp

#include <memory>
#include <vector>
#include "core/NonCopyable.hpp"
#include "backend/vulkan/component/VulkanInstance.hpp"
#include "backend/vulkan/vulkan/vulkan_wrapper.h"

namespace MNN {
class VulkanDevice : public NonCopyable {
public:
    explicit VulkanDevice(std::shared_ptr<VulkanInstance> instance);
    explicit VulkanDevice(std::shared_ptr<VulkanInstance> instance, VkPhysicalDevice physicalDevice, VkDevice device,
                          uint32_t queueFamilyIndex, VkQueue queue);
    virtual ~VulkanDevice();

    const VkDevice get() const {
        return mDevice;
    }

    void getDeviceQueue(const uint32_t familyIndex, const uint32_t queueIndex, VkQueue& queue);
    const VkQueue acquireDefaultDevQueue() const;

    // VkBuffer/VkDeviceMemory
    const VkResult createBuffer(VkBuffer& buffer, const size_t size, const VkBufferUsageFlags usage,
                                const VkSharingMode shared, const VkAllocationCallbacks* allocator = nullptr) const;
    const void getBufferMemoryRequirements(const VkBuffer buffer, VkMemoryRequirements& memoryRequirements) const;
    const VkResult allocMemory(VkDeviceMemory& memory, const VkMemoryAllocateInfo& allocateInfo,
                               const VkAllocationCallbacks* allocator = nullptr) const;
    const void freeMemory(const VkDeviceMemory& memory, const VkAllocationCallbacks* allocator = nullptr) const;
    const VkResult mapMemory(const VkDeviceMemory memory, const VkDeviceSize offset, const VkDeviceSize size,
                             const VkMemoryMapFlags flags, void** ppData) const;
    const void unmapMemory(const VkDeviceMemory memory) const;
    const VkResult bindBufferMemory(const VkBuffer buffer, const VkDeviceMemory memory,
                                    const VkDeviceSize memoryOffset = 0) const;
    const void destroyBuffer(const VkBuffer buffer, const VkAllocationCallbacks* allocator = nullptr) const;
    const VkResult flushMappedMemoryRanges(const VkMappedMemoryRange* memoryRanges,
                                           const uint32_t memoryRangeCount = 1) const;
    const VkResult invalidateMappedMemoryRanges(const VkMappedMemoryRange* memoryRanges,
                                                const uint32_t memoryRangeCount = 1) const;

    // VkCommand*
    const VkResult createCommandPool(
        VkCommandPool& cmdPool, const VkCommandPoolCreateFlags flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        const VkAllocationCallbacks* allocator = nullptr) const;
    const void destroyCommandPool(const VkCommandPool& cmdPool, const VkAllocationCallbacks* allocator = nullptr) const;
    const VkResult allocateCommandBuffers(const VkCommandPool& cmdPool, VkCommandBuffer* cmdBuffers,
                                          const uint32_t cmdBufferCount    = 1,
                                          const VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) const;
    const void freeCommandBuffers(const VkCommandPool& cmdPool, const VkCommandBuffer* cmdBuffers,
                                  const uint32_t cmdBufferCount = 1) const;
    const VkResult allocateCommandBuffer(const VkCommandPool& cmdPool, VkCommandBuffer& cmdBuffer,
                                         const VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) const;
    const void freeCommandBuffer(const VkCommandPool& cmdPool, const VkCommandBuffer& cmdBuffer) const;

    // VkFence
    const VkResult createFence(VkFence& fence, const VkAllocationCallbacks* allocator = nullptr) const;
    const VkResult waitForFence(const VkFence& fence, const uint64_t timeout) const;
    const VkResult waitForFences(const uint32_t fenceCount, const VkFence* fences, const VkBool32 waitAll,
                                 const uint64_t timeout) const;
    void destroyFence(const VkFence& fence, const VkAllocationCallbacks* allocator = nullptr) const;
    const VkResult resetFences(const uint32_t fenceCount, const VkFence* fences) const;
    const VkResult resetFence(const VkFence& fence) const;

    // VkSemaphore
    const VkResult createSemaphore(VkSemaphore& semaphore, const VkAllocationCallbacks* allocator = nullptr) const;
    const void destroySemaphore(const VkSemaphore& semaphore, const VkAllocationCallbacks* allocator = nullptr) const;

    // VkImage/VkSampler
    const VkResult createImage(VkImage& image, const VkImageType imageType, const uint32_t width, const uint32_t height,
                               const uint32_t depth, const VkFormat format, VkImageUsageFlags usage,
                               const VkAllocationCallbacks* allocator = nullptr) const;
    const void destroyImage(const VkImage& image, const VkAllocationCallbacks* allocator = nullptr) const;

    const void getImageMemoryRequirements(const VkImage& image, VkMemoryRequirements& memoryRequirements) const;

    const void bindImageMemory(const VkImage& image, const VkDeviceMemory& memory,
                               const VkDeviceSize& memoryOffset = 0) const;

    const VkResult createImageView(VkImageView& view, const VkImage& image, const VkImageViewType& viewType,
                                   const VkFormat& format, const VkAllocationCallbacks* allocator = nullptr) const;
    const void destroyImageView(const VkImageView& imageView, const VkAllocationCallbacks* allocator = nullptr) const;

    const VkResult createSampler(VkSampler& sampler, const VkFilter& filter = VK_FILTER_NEAREST,
                                 const VkSamplerAddressMode& mode       = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                 const VkAllocationCallbacks* allocator = nullptr) const;

    const void destroySampler(const VkSampler& sampler, const VkAllocationCallbacks* allocator = nullptr) const;

    //  VkPipeline
    const VkResult createPipelineCache(VkPipelineCache& pipelineCache,
                                       const VkAllocationCallbacks* allocator = nullptr) const;
    const void destroyPipelineCache(const VkPipelineCache& pipelineCache,
                                    const VkAllocationCallbacks* allocator = nullptr) const;

    const VkResult createShaderModule(VkShaderModule& shaderModule, const size_t codeSize, const uint32_t* pCode,
                                      const VkAllocationCallbacks* allocator = nullptr) const;
    const void destroyShaderModule(const VkShaderModule& shaderModule,
                                   const VkAllocationCallbacks* allocator = nullptr) const;

    const void updateDescriptorSets(uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites,
                                    uint32_t descriptorCopyCount, const VkCopyDescriptorSet* pDescriptorCopies) const;
    const void updateWriteDescriptorSet(const VkWriteDescriptorSet& descriptorWrite) const;

    const VkResult createDescriptorSetLayout(VkDescriptorSetLayout& setLayout, const uint32_t bindingCount,
                                             const VkDescriptorSetLayoutBinding* bindings,
                                             const VkAllocationCallbacks* allocator = nullptr) const;
    const VkResult createPipelineLayout(VkPipelineLayout& pipelineLayout, const VkDescriptorSetLayout& setLayout,
                                        const VkAllocationCallbacks* allocator = nullptr) const;
    const void destroyPipelineLayout(const VkPipelineLayout& pipelineLayout,
                                     const VkAllocationCallbacks* allocator = nullptr) const;
    const VkResult createComputePipelines(VkPipeline* pipelines, const VkComputePipelineCreateInfo* createInfos,
                                          const uint32_t createInfoCount, const VkPipelineCache& pipelineCache,
                                          const VkAllocationCallbacks* allocator = nullptr) const;
    const VkResult createComputePipeline(VkPipeline& pipeline, const VkShaderModule& shaderMoule,
                                         const VkPipelineLayout& pipelineLayout, const VkPipelineCache& pipelineCache,
                                         const VkSpecializationInfo* pSpecializationInfo = nullptr,
                                         const VkAllocationCallbacks* allocator          = nullptr) const;

    const void destroyDescriptorSetLayout(const VkDescriptorSetLayout& descriptorSetLayout,
                                          const VkAllocationCallbacks* allocator = nullptr) const;

    const void destroyPipeline(const VkPipeline& pipeline, const VkAllocationCallbacks* allocator = nullptr) const;

    const VkResult createDescriptorPool(VkDescriptorPool& descriptorPool, const uint32_t poolSizeCount,
                                        const VkDescriptorPoolSize* pPoolSizes,
                                        const VkAllocationCallbacks* allocator = nullptr) const;

    const VkResult allocateDescriptorSet(VkDescriptorSet& pDescriptorSet, const VkDescriptorPool& descPool,
                                         const VkDescriptorSetLayout& setLayout) const;

    const VkResult freeDescriptorSets(const VkDescriptorPool& descriptorPool, const uint32_t descriptorSetCount,
                                      const VkDescriptorSet* pDescriptorSets) const;

    const void destroyDescriptorPool(const VkDescriptorPool& descriptorPool,
                                     const VkAllocationCallbacks* allocator = nullptr) const;

    const VkPhysicalDeviceProperties& proty() const {
        return mDeviceProty;
    }
    const VkPhysicalDeviceMemoryProperties& memProty() const {
        return mMemoryProty;
    }

    const bool success() const {
        return (VK_NULL_HANDLE != mDevice);
    }
    
    const float getTimestampPeriod() const {
        return mDeviceProty.limits.timestampPeriod;
    }
    
    const int getMaxComputeWorkGroupInvocations() const {
        return mDeviceProty.limits.maxComputeWorkGroupInvocations;
    }
    const int32_t getLocalMemorySize() const {
        return mLocalMemorySize;
    }
    
    const void getMaxComputeWorkGroupSize(std::vector<int> &groups) const{
        if(groups.size() == 3){
            groups[0] = mDeviceProty.limits.maxComputeWorkGroupSize[0];
            groups[1] = mDeviceProty.limits.maxComputeWorkGroupSize[1];
            groups[2] = mDeviceProty.limits.maxComputeWorkGroupSize[2];
        }
    }

    uint32_t getSubgroupSize() const {
        return mSubgroupSize;
    }

    bool getFP16Support() const {
        return mFP16Info.supportFP16;
    }

private:
    // Set mFP16Info
    void checkFP16(const std::vector<VkExtensionProperties>& availableExts);
    // Set mCoopMatInfo
    void checkCoopMat(const std::vector<VkExtensionProperties>& availableExts);


private:
    bool mOwner;
    std::shared_ptr<VulkanInstance> mInstance; ///< refer to Instance object used to create device
    uint32_t mQueueFamilyIndex;
    VkPhysicalDevice mPhysicalDevice;
    VkDevice mDevice;
    VkPhysicalDeviceProperties mDeviceProty;
    VkQueue mQueue;
    VkPhysicalDeviceMemoryProperties mMemoryProty;
    uint32_t mSubgroupSize;
    uint32_t mLocalMemorySize = 0;

// FP16 related
private:
struct FP16Info {
    bool supportFP16{false};
    bool FP16FromExtension{false};
    VkPhysicalDeviceVulkan11Features enabledVulkan11Features{};
    VkPhysicalDeviceVulkan12Features enabledVulkan12Features{};
    VkPhysicalDeviceShaderFloat16Int8Features enabledShaderFloat16Int8Features{};
    VkPhysicalDevice16BitStorageFeatures enabled16BitStorageFeatures{};
};
    FP16Info mFP16Info{};

// CoopMat related
public:
    struct CoopMatInfo {
        bool supportCoopMat{false};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR enabledCoopMatFeatures{};
        std::vector<std::vector<uint32_t>> fp32CoopMatShape;
        std::vector<std::vector<uint32_t>> fp16CoopMatShape;
        std::vector<uint32_t> selectedFP32CoopMatShape; // {M, N, K}
        std::vector<uint32_t> selectedFP16CoopMatShape; // {M, N, K}
    };
private:
    CoopMatInfo mCoopMatInfo{};
public:
    CoopMatInfo getCoopMatInfo() const {
        return mCoopMatInfo;
    }
};
} // namespace MNN
#endif /* VulkanDevice_hpp */
