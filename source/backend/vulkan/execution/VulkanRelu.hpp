//
//  VulkanRelu.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanRelu_hpp
#define VulkanRelu_hpp

#include <stdio.h>

#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanRelu : public VulkanBasicExecution {
public:
    VulkanRelu(Backend* bn, float slope);
    virtual ~VulkanRelu();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    float mSlope;
    std::shared_ptr<VulkanBuffer> mGpuReluParam;
    const VulkanPipeline* mReluPipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDescriptorSet;
};

class VulkanRelu6 : public VulkanBasicExecution {
public:
    VulkanRelu6(Backend* bn);
    virtual ~VulkanRelu6();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mGpuRelu6Param;
    const VulkanPipeline* mRelu6Pipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDescriptorSet;
};

class VulkanPrelu : public VulkanBasicExecution {
public:
    VulkanPrelu(Backend* bn, const Op* op);
    virtual ~VulkanPrelu();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mGpuPreluParam;
    std::shared_ptr<VulkanImage> mSlope;
    const VulkanPipeline* mPreluPipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDescriptorSet;
};

} // namespace MNN

#endif /* VulkanRelu_hpp */
