//
//  VulkanArgMax.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanArgMax_hpp
#define VulkanArgMax_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"
#include "VulkanRaster.hpp"

namespace MNN {
class VulkanArgMax : public VulkanBasicExecution {
public:
    VulkanArgMax(const Op* op, Backend* bn);
    virtual ~VulkanArgMax();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    const VulkanPipeline* mArgmaxPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    int mAxis;
};

} // namespace MNN

#endif /* VulkanArgMax_hpp */
