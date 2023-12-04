//
//  VulkanGridSample.hpp
//  MNN
//
//  Created by MNN on 2021/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanGridSample_hpp
#define VulkanGridSample_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"

namespace MNN {
class VulkanGridSample : public VulkanBasicExecution {
public:
    VulkanGridSample(const Op* op, Backend* bn);
    virtual ~VulkanGridSample();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    bool mAlignCorners;
    std::shared_ptr<VulkanBuffer> mGridSampleParam;
    const VulkanPipeline* mGridSamplePipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
};
} // namespace MNN

#endif /* VulkanGridSample_hpp */
