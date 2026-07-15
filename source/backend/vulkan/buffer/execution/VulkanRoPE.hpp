#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef VulkanRoPE_hpp
#define VulkanRoPE_hpp

#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanRoPE : public VulkanBasicExecution {
public:
    VulkanRoPE(const Op* op, Backend* backend);
    ~VulkanRoPE() override;

    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;
    bool onClone(Backend* backend, const Op* op, VulkanBasicExecution** dst) override;

private:
    struct GpuParam {
        ivec4 size0; // seqLen, headDim, numHead, kvNumHead
        ivec4 size1; // ropeHalfDim, qNorm, kNorm, 0
        vec4 eps;    // qEps, kEps, 0, 0
    };

    VulkanRoPE(Backend* backend, const VulkanRoPE* source);
    bool prepareGamma(const LayerNorm* norm, std::shared_ptr<Tensor>& gamma, float& eps);
    void createPipeline();

    int mRopeCutHeadDim = 0;
    int mNumHead = 0;
    int mKvNumHead = 0;
    int mHeadDim = 0;
    float mQEps = 0.0f;
    float mKEps = 0.0f;
    bool mQNorm = false;
    bool mKNorm = false;
    bool mValid = false;
    bool mUseFP16 = false;
    std::shared_ptr<Tensor> mQGamma;
    std::shared_ptr<Tensor> mKGamma;
    std::shared_ptr<VulkanBuffer> mParam;
    const VulkanPipeline* mPipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
};

} // namespace MNN

#endif // VulkanRoPE_hpp

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
