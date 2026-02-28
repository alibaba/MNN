//
//  VulkanLinearAttention.hpp
//  MNN
//
//  Created by MNN on 2026/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanLinearAttention_hpp
#define VulkanLinearAttention_hpp

#include "VulkanBasicExecution.hpp"

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {

class VulkanLinearAttention : public VulkanBasicExecution {
public:
    VulkanLinearAttention(const MNN::Op* op, Backend* backend);
    virtual ~VulkanLinearAttention();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;
    virtual bool onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) override;

private:
    std::string mAttentionType;
    int mNumKHeads;
    int mNumVHeads;
    int mHeadKDim;
    int mHeadVDim;
    bool mUseQKL2Norm;

    // Persistent state buffers (STATIC)
    std::shared_ptr<Tensor> mConvState;       // [B, D, K-1]
    std::shared_ptr<Tensor> mRecurrentState;  // [B, H, d_k, d_v]

    // Temporary buffer (DYNAMIC)
    std::shared_ptr<Tensor> mConvOut;         // [B, D, L]

    bool mFirstResize = true;

    // Pipelines
    const VulkanPipeline* mConvSiluPipeline;
    const VulkanPipeline* mConvStateUpdatePipeline;
    const VulkanPipeline* mGatedDeltaRulePipeline;

    // Descriptor sets
    std::shared_ptr<VulkanLayout::DescriptorSet> mConvSiluDesSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mConvStateUpdateDesSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mGatedDeltaRuleDesSet;

    // Uniform buffers
    std::shared_ptr<VulkanBuffer> mConvSiluParam;
    std::shared_ptr<VulkanBuffer> mConvStateUpdateParam;
    std::shared_ptr<VulkanBuffer> mGatedDeltaRuleParam;
};

} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* VulkanLinearAttention_hpp */
