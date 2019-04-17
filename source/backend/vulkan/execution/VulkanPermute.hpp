//
//  VulkanPermute.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanPermute_hpp
#define VulkanPermute_hpp
#include "VulkanBasicExecution.hpp"
#include "VulkanImageConverter.hpp"
namespace MNN {
class VulkanPermute : public VulkanBasicExecution {
public:
    VulkanPermute(const Op* op, Backend* bn);
    virtual ~VulkanPermute();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mParamBuffer;
    const VulkanPipeline* mVulkanPermutePipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDescriptorSet;
    std::vector<int> mDims;
    Tensor mTempSource;
    Tensor mTempDest;
    std::shared_ptr<VulkanImageConverter> mSourceTransform;
    std::shared_ptr<VulkanImageConverter> mDestTransform;
};
} // namespace MNN
#endif
