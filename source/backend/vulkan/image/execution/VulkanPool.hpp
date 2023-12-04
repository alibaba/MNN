//
//  VulkanPool.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanPool_hpp
#define VulkanPool_hpp

#include "VulkanBasicExecution.hpp"
namespace MNN {
class VulkanPool : public VulkanBasicExecution {
public:
    VulkanPool(const Op* op, Backend* bn);
    virtual ~VulkanPool();

    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    const VulkanPipeline* mPoolPipeline;
    const Pool* mCommon;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
};
} // namespace MNN
#endif /* VulkanPool_hpp */
