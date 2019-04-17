//
//  VulkanSoftmax.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanSoftmax_hpp
#define VulkanSoftmax_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"

namespace MNN {
class VulkanSoftmax : public VulkanBasicExecution {
public:
    VulkanSoftmax(const Op* op, Backend* bn);
    virtual ~VulkanSoftmax();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    const VulkanPipeline* mSoftmaxPipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDescriptorSet;
    int mAxis;
    const VulkanBackend* mVkBackend;
};
} // namespace MNN

#endif /* VulkanSoftmax_hpp */
