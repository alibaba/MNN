//
//  VulkanNormlize.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanNormlize_hpp
#define VulkanNormlize_hpp
#include "VulkanBasicExecution.hpp"

namespace MNN {
class VulkanNormlize : public VulkanBasicExecution {
public:
    VulkanNormlize(const Op* op, Backend* bn);
    virtual ~VulkanNormlize();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mParamBuffer;
    const VulkanPipeline* mVulkanNormlizePipeline;
    const VulkanPipeline* mVulkanScalePipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mNormlizeDescriptorSet;
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
