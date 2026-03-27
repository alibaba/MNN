#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef VulkanAttention_hpp
#define VulkanAttention_hpp

#include "VulkanBasicExecution.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {

class VulkanAttention : public VulkanBasicExecution {
public:
    VulkanAttention(const Op* op, Backend* bn);
    ~VulkanAttention() override;

    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;
    ErrorCode onBeforeExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    bool onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) override;

private:
    struct GpuParam {
        ivec4 s0; // qLen, kLen, headNum, kvHeadNum
        ivec4 s1; // headDim, group, pastLen, totalLen
        ivec4 s2; // maskQlen, maskKvlen, hasMask, cacheMaxLen
        vec4 f0;  // scale, 0, 0, 0
    };

    struct KVCache {
        int maxLen = 0;
        int kvHeadNum = 0;
        int headDim = 0;
        int expandChunk = 64;
        bool fp16 = false;
        std::shared_ptr<VulkanBuffer> key;
        std::shared_ptr<VulkanBuffer> value;

        void reset();
        void ensureCapacity(VulkanBackend* vkBn, int requiredLen, int kvH, int d, bool useFP16);
    };

    const Op* mOp = nullptr;
    bool mNeedKvCache = false;
    bool mUseFP16 = false;
    KVMeta* mMeta = nullptr;

    int mQueryLen = 0;
    int mKeyLen = 0;
    int mHeadNum = 0;
    int mKvHeadNum = 0;
    int mHeadDim = 0;

    std::shared_ptr<KVCache> mKVCache;

    std::shared_ptr<VulkanBuffer> mParam;
    const VulkanPipeline* mAttentionPipeline = nullptr;
    const VulkanPipeline* mAttentionLegacyPipeline = nullptr;
    const VulkanPipeline* mDecodeQ1SubgroupPipeline = nullptr;
    const VulkanPipeline* mDecodeQ1SubgroupHD128Pipeline = nullptr;
    const VulkanPipeline* mUpdatePipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mAttentionSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mAttentionLegacySet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDecodeQ1SubgroupSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDecodeQ1SubgroupHD128Set;
    std::shared_ptr<VulkanLayout::DescriptorSet> mUpdateSet;

    bool mUsePrefill = false;
    int mPrefillTotalLen = 0; // encoded totalLen for prefill multi-pass
    int mQueryLen4 = 0; // padded qLen for rearranged Qtmp (multiple of 4)
    std::shared_ptr<Tensor> mTempQuery;

    // Prefill K-block mode temporaries (avoid O(qLen*totalLen) intermediates).
    std::shared_ptr<Tensor> mTempQKBlock;
    std::shared_ptr<Tensor> mTempWBlock;
    std::shared_ptr<Tensor> mTempM;
    std::shared_ptr<Tensor> mTempL;
    std::shared_ptr<Tensor> mTempAlpha;
    std::shared_ptr<Tensor> mTempOAcc;

    const VulkanPipeline* mRearrangeQPipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mRearrangeQSet;

    const VulkanPipeline* mInitStatePipeline = nullptr;
    const VulkanPipeline* mQKBlockFullPipeline = nullptr;
    const VulkanPipeline* mQKBlockPipeline = nullptr;
    const VulkanPipeline* mSoftmaxOnlinePipeline = nullptr;
    const VulkanPipeline* mQKVAccFullPipeline = nullptr;
    const VulkanPipeline* mQKVAccPipeline = nullptr;
    const VulkanPipeline* mFinalizePipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mInitStateSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mQKBlockFullSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mQKBlockSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mSoftmaxOnlineSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mQKVAccFullSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mQKVAccSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mFinalizeSet;

    uint32_t mSoftmaxOnlineLocalSize = 0;
    uint32_t mDecodeQ1SubgroupLocalSize = 0;
};

} // namespace MNN

#endif // VulkanAttention_hpp

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
