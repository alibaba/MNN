#ifndef VulkanRoPE_hpp
#define VulkanRoPE_hpp

#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanRoPE : public VulkanBasicExecution {
public:
    VulkanRoPE(const Op* op, Backend* backend);
    virtual ~VulkanRoPE();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;
    virtual bool onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) override;

private:
    VulkanRoPE(Backend* backend, int ropeCutHeadDim, std::shared_ptr<VulkanBuffer> qGamma, float qEps,
               std::shared_ptr<VulkanBuffer> kGamma, float kEps);
    void initPipeline();
    void parseAttrs(const Op* op);
    bool useNorm() const;

private:
    int mRopeCutHeadDim = 0;
    std::shared_ptr<VulkanBuffer> mQGamma;
    std::shared_ptr<VulkanBuffer> mKGamma;
    float mQEps = 0.0f;
    float mKEps = 0.0f;
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    const VulkanPipeline* mPipeline = nullptr;
    const VulkanPipeline* mSubgroupPipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mSubgroupDescriptorSet;
    uint32_t mSubgroupSize = 0;
};

} // namespace MNN

#endif // VulkanRoPE_hpp
