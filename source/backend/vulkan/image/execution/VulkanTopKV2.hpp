//
//  VulkanTopKV2.hpp
//  MNN
//
//  Vulkan image-mode implementation of TopKV2.
//

#ifndef VulkanTopKV2_hpp
#define VulkanTopKV2_hpp

#include "VulkanBasicExecution.hpp"

namespace MNN {
class VulkanTopKV2 : public VulkanBasicExecution {

public:
    VulkanTopKV2(const Op* op, Backend* bn, int k);
    virtual ~VulkanTopKV2();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    const VulkanPipeline* mPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    std::shared_ptr<VulkanBuffer> mGpuParam;
    int mK;
    bool mLargest;
};

} // namespace MNN

#endif /* VulkanTopKV2_hpp */