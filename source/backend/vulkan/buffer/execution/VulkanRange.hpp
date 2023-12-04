//
//  VulkanRange.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanRange_hpp
#define VulkanRange_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanRange : public VulkanBasicExecution {
public:
    VulkanRange(halide_type_t type, Backend* bn);
    virtual ~VulkanRange();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mParam;
    const VulkanPipeline* mPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDesSet;
};

} // namespace MNN

#endif /* VulkanRange_hpp */
