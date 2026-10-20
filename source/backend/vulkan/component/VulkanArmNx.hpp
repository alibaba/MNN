// Experimental ARM NX path through single-op TOSA data graphs. No VGF or ARM runtime dependency.
#ifndef MNN_VULKAN_ARM_NX_HPP
#define MNN_VULKAN_ARM_NX_HPP
#ifdef MNN_VULKAN_ARM_NX

#include "VulkanDevice.hpp"
#include <array>

namespace MNN {

class VulkanArmNx : public NonCopyable {
public:
    // A:[batch,M,K], B:[batch,K,N], C:[batch,M,N]. FP32->FP32 or INT8->INT32, zero points 0.
    static std::unique_ptr<VulkanArmNx> createMatMul(const VulkanDevice& device, uint32_t m, uint32_t k, uint32_t n,
                                                     bool int8 = false, uint32_t batch = 1);
    ~VulkanArmNx();
    // Buffer ranges must contain the packed, four-byte-padded payload. Transfers stay on the GPU.
    void encode(VkCommandBuffer cmd, VkBuffer a, VkDeviceSize aOffset, VkBuffer b, VkDeviceSize bOffset,
                VkBuffer output, VkDeviceSize outputOffset) const;
    static std::vector<uint32_t> matMulSpirv(uint32_t m, uint32_t k, uint32_t n, bool int8, uint32_t batch = 1);
    static uint64_t recordedDispatchCount();

private:
    explicit VulkanArmNx(const VulkanDevice& device) : mDevice(device) {}
    bool init(uint32_t m, uint32_t k, uint32_t n, bool int8, uint32_t batch);
    bool allocate(const VkMemoryRequirements& requirements, VkDeviceMemory& memory);
    bool createTensor(size_t index, uint32_t batch, int64_t rows, int64_t columns, VkFormat format);
    void barrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                 VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) const;
    struct Resource {
        VkTensorARM tensor = VK_NULL_HANDLE;
        VkTensorViewARM view = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize bytes = 0;
        std::array<int64_t, 3> dimensions{};
        VkTensorDescriptionARM description{};
    };
    const VulkanDevice& mDevice;
    std::array<Resource, 3> mResources{};
    VkDescriptorSetLayout mSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout mLayout = VK_NULL_HANDLE;
    VkDescriptorPool mDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet mSet = VK_NULL_HANDLE;
    VkShaderModule mShader = VK_NULL_HANDLE;
    VkPipeline mPipeline = VK_NULL_HANDLE;
    VkDataGraphPipelineSessionARM mSession = VK_NULL_HANDLE;
    std::vector<VkDeviceMemory> mSessionMemory;
    PFN_vkCreateTensorARM mCreateTensor = nullptr;
    PFN_vkDestroyTensorARM mDestroyTensor = nullptr;
    PFN_vkGetTensorMemoryRequirementsARM mGetTensorMemoryRequirements = nullptr;
    PFN_vkBindTensorMemoryARM mBindTensorMemory = nullptr;
    PFN_vkCreateTensorViewARM mCreateTensorView = nullptr;
    PFN_vkDestroyTensorViewARM mDestroyTensorView = nullptr;
    PFN_vkCreateDataGraphPipelinesARM mCreatePipelines = nullptr;
    PFN_vkCreateDataGraphPipelineSessionARM mCreateSession = nullptr;
    PFN_vkDestroyDataGraphPipelineSessionARM mDestroySession = nullptr;
    PFN_vkGetDataGraphPipelineSessionBindPointRequirementsARM mGetSessionBindPoints = nullptr;
    PFN_vkGetDataGraphPipelineSessionMemoryRequirementsARM mGetSessionMemoryRequirements = nullptr;
    PFN_vkBindDataGraphPipelineSessionMemoryARM mBindSessionMemory = nullptr;
    PFN_vkCmdDispatchDataGraphARM mDispatch = nullptr;
    PFN_vkCmdPipelineBarrier2 mBarrier = nullptr;
};
} // namespace MNN
#endif
#endif
