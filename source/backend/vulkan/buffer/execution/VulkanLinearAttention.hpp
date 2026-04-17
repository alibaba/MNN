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

struct KVMeta;

struct VulkanLinearAttentionState {
    std::shared_ptr<Tensor> mConvState;
    std::shared_ptr<Tensor> mRecurrentState;
    int mBatch = 0;
    int mConvDim = 0;
    int mConvStateSize = 0;
    int mNumVHeads = 0;
    int mHeadKDim = 0;
    int mHeadVDim = 0;
};

class VulkanLinearAttention : public VulkanBasicExecution {
public:
    VulkanLinearAttention(const MNN::Op* op, Backend* backend);
    virtual ~VulkanLinearAttention();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;
    virtual bool onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) override;

private:
    ErrorCode ensurePersistentState(VulkanBackend* vkBn, int batch, int convDim, int convStateSize);
    ErrorCode resetPersistentState(VulkanBackend* vkBn);

private:
    std::string mAttentionType;
    int mNumKHeads = 0;
    int mNumVHeads = 0;
    int mHeadKDim = 0;
    int mHeadVDim = 0;
    bool mUseQKL2Norm = false;
    bool mUseFP16 = false;
    uint32_t mSubgroupSize = 0;
    uint32_t mSubgroupsPerWorkgroup = 4;
    KVMeta* mMeta = nullptr;

    std::shared_ptr<VulkanLinearAttentionState> mStateCache;

    std::shared_ptr<Tensor> mConvOut;
    std::shared_ptr<Tensor> mQ;
    std::shared_ptr<Tensor> mK;
    std::shared_ptr<Tensor> mV;

    const VulkanPipeline* mConvSiluPipeline = nullptr;
    const VulkanPipeline* mConvStateUpdatePipeline = nullptr;
    const VulkanPipeline* mQKVPrepPipeline = nullptr;
    const VulkanPipeline* mPrefillPipeline = nullptr;
    const VulkanPipeline* mDecodePipeline = nullptr;

    std::shared_ptr<VulkanLayout::DescriptorSet> mConvSiluDesSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mConvStateUpdateDesSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mQKVPrepDesSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mPrefillDesSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDecodeDesSet;

    std::shared_ptr<VulkanBuffer> mConvSiluParam;
    std::shared_ptr<VulkanBuffer> mConvStateUpdateParam;
    std::shared_ptr<VulkanBuffer> mQKVPrepParam;
    std::shared_ptr<VulkanBuffer> mPrefillParam;
    std::shared_ptr<VulkanBuffer> mDecodeParam;
};

} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* VulkanLinearAttention_hpp */
