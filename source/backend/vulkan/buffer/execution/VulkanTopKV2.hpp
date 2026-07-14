//
//  VulkanTopKV2.hpp
//  MNN
//
//  Vulkan buffer-mode implementation of TopKV2.
//

#ifndef VulkanTopKV2_hpp
#define VulkanTopKV2_hpp

#include "VulkanBasicExecution.hpp"
#include <string>

namespace MNN {
class VulkanTopKV2 : public VulkanBasicExecution {
public:
    VulkanTopKV2(const Op* op, Backend* bn, Tensor* input, Tensor* output);
    virtual ~VulkanTopKV2();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    bool _createPipeline(int k, Tensor* input);

private:
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    const VulkanPipeline* mPipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    std::string mPipelineName;
    int mK = 0;
    int mTop1ThreadNumber = 0;
    bool mLargest = true;
};

} // namespace MNN

#endif /* VulkanTopKV2_hpp */
