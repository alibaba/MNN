//
//  VulkanUnary.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanUnary_hpp
#define VulkanUnary_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanUnary : public VulkanBasicExecution {
public:
    VulkanUnary(const Op* op, Backend* bn);
    virtual ~VulkanUnary();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    const MNN::Op* mOp;
    std::shared_ptr<VulkanBuffer> mParam;
    const VulkanPipeline* mUnaryPipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDesSet;
};

} // namespace MNN

#endif /* VulkanUnary_hpp */
