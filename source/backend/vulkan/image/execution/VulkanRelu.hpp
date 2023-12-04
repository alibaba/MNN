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
    VulkanRelu(Backend* bn, const Op* op);
    virtual ~VulkanRelu();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    float mSlope[4];
    std::vector<std::shared_ptr<VulkanBuffer>> mGpuReluParam;
    const VulkanPipeline* mReluPipeline;
    std::vector<std::shared_ptr<VulkanLayout::DescriptorSet>> mDescriptorSet;
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
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
};

} // namespace MNN

#endif /* VulkanRelu_hpp */
