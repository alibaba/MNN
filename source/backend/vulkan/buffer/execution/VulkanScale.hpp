//
//  VulkanScale.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanScale_hpp
#define VulkanScale_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"

namespace MNN {
class VulkanScale : public VulkanBasicExecution {
public:
    VulkanScale(const Op* op, Backend* bn);
    virtual ~VulkanScale();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mScaleParam;
    const VulkanPipeline* mScalePipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    std::shared_ptr<VulkanBuffer> mScaleBuffer;
    std::shared_ptr<VulkanBuffer> mBiasBuffer;
    const VulkanSampler* mSampler;
};
} // namespace MNN

#endif /* VulkanScale_hpp */
