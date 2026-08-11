//
//  VulkanRange.hpp
//  MNN
//
//  Created by MNN on 2026/06/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanRange_hpp
#define VulkanRange_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"

namespace MNN {
class VulkanRange : public VulkanBasicExecution {
public:
    VulkanRange(Backend *backend);
    virtual ~VulkanRange();
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                               const VulkanCommandPool::Buffer *cmdBuffer) override;

private:
    std::vector<std::shared_ptr<VulkanBuffer>> mParams;
    const VulkanPipeline* mPipeline;
    std::vector<std::shared_ptr<VulkanLayout::DescriptorSet>> mDescriptorSets;
};
}

#endif // VulkanRange_hpp