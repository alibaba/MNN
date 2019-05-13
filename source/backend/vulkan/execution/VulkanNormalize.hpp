//
//  VulkanNormalize.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanNormalize_hpp
#define VulkanNormalize_hpp
#include "VulkanBasicExecution.hpp"

namespace MNN {
class VulkanNormalize : public VulkanBasicExecution {
public:
    VulkanNormalize(const Op* op, Backend* bn);
    virtual ~VulkanNormalize();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mParamBuffer;
    const VulkanPipeline* mVulkanNormalizePipeline;
    const VulkanPipeline* mVulkanScalePipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mNormalizeDescriptorSet;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mScaleDescriptorSet;
    std::shared_ptr<VulkanBuffer> mScale;
    std::shared_ptr<VulkanBuffer> mBias;
    float mEps;
    Tensor mTempTensor;
    const VulkanSampler* mSampler;
    const VulkanBackend* mVkBackend;
};
} // namespace MNN

#endif
