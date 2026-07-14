#ifndef VulkanSharedGather_hpp
#define VulkanSharedGather_hpp

#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanSharedGather : public VulkanBasicExecution {
public:
    VulkanSharedGather(VulkanBackend* backend, int ci, int co, int quantBits, uint32_t padN, uint32_t blockSize,
                       uint32_t blockStride, uint32_t weightStride, bool offsetZero,
                       std::shared_ptr<VulkanBuffer> weight, std::shared_ptr<VulkanBuffer> meta);
    virtual ~VulkanSharedGather();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;
    virtual bool onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) override;

private:
    int mCi = 0;
    int mCo = 0;
    int mQuantBits = 0;
    uint32_t mPadN = 0;
    uint32_t mBlockSize = 1;
    uint32_t mBlockStride = 1;
    uint32_t mWeightStride = 0;
    uint32_t mOffsetZero = 0;
    uint32_t mLocalSize = 64;
    std::shared_ptr<VulkanBuffer> mWeight;
    std::shared_ptr<VulkanBuffer> mMeta;
    const VulkanPipeline* mPipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
};

} // namespace MNN

#endif // VulkanSharedGather_hpp
