//
//  VulkanSelect.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanSelect_hpp
#define VulkanSelect_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanSelect : public VulkanBasicExecution {
public:
    VulkanSelect(const Op* op, Backend* bn);
    virtual ~VulkanSelect();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mParam;
    const VulkanPipeline* mPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDesSet;
};

} // namespace MNN

#endif /* VulkanSelect_hpp */
