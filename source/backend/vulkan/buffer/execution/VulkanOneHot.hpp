//
//  VulkanOneHot.hpp
//  MNN
//
//  Created by MNN on 2020/07/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanOneHot_hpp
#define VulkanOneHot_hpp

#include "VulkanBasicExecution.hpp"
namespace MNN {
class VulkanOneHot : public VulkanBasicExecution {
public:
    VulkanOneHot(const Op* op, Backend* bn);
    virtual ~VulkanOneHot();

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    const VulkanPipeline* mPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    int mAxis;
};
} // namespace MNN
#endif /* VulkanPool_hpp */
