//
//  VulkanTopKV2.hpp
//  MNN
//
//  Vulkan buffer-mode implementation of TopKV2.
//

#ifndef VulkanTopKV2_hpp
#define VulkanTopKV2_hpp

#include "VulkanBasicExecution.hpp"

namespace MNN {
class VulkanTopKV2 : public VulkanBasicExecution {
public:
    VulkanTopKV2(const Op* op, Backend* bn, int k, Tensor* input);
    virtual ~VulkanTopKV2();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    const VulkanPipeline* mPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    int mK;
    bool mLargest;
};

} // namespace MNN

#endif /* VulkanTopKV2_hpp */
