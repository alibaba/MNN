#ifndef MNN_VULKAN_CONV1X1_ARM_NX_HPP
#define MNN_VULKAN_CONV1X1_ARM_NX_HPP
#ifdef MNN_VULKAN_ARM_NX
#include "VulkanBasicExecution.hpp"
#include "VulkanArmNx.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {
// Experimental W4/W8 + dynamically quantized A8 Linear, for both prefill and decode.
class VulkanConv1x1ArmNx : public VulkanBasicExecution {
public:
    static VulkanBasicExecution* create(VulkanBackend* backend, const Convolution2DCommon* common, const float* bias,
                                        int ci, int co, std::shared_ptr<ConvolutionCommon::Int8Common> quant);
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmd) override;
    bool onClone(Backend* backend, const Op* op, VulkanBasicExecution** dst) override;

private:
    struct Resource {
        uint32_t ci = 0, co = 0, blocks = 0, blockSize = 0, activation = 0;
        VkDevice device = VK_NULL_HANDLE;
        const Convolution2DCommon* common = nullptr;
        std::shared_ptr<ConvolutionCommon::Int8Common> quant;
        std::shared_ptr<VulkanBuffer> weight, meta, bias;
        std::vector<float> biasHost;
    };
    VulkanConv1x1ArmNx(VulkanBackend* backend, std::shared_ptr<Resource> resource);
    std::shared_ptr<Resource> mResource;
    std::unique_ptr<VulkanBasicExecution> mFallback;
    std::unique_ptr<VulkanArmNx> mArmNx;
    std::shared_ptr<VulkanBuffer> mActivation, mActivationMeta, mAccumulator;
    const VulkanPipeline* mQuantize = nullptr;
    const VulkanPipeline* mDequantize = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mQuantizeSet, mDequantizeSet;
    uint32_t mRows = 0;
};
} // namespace MNN
#endif
#endif
