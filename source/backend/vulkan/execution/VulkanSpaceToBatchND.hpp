//
//  VulkanSpaceToBatchND.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanSpaceToBatchND_hpp
#define VulkanSpaceToBatchND_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanSpaceToBatchND : public VulkanBasicExecution {
public:
    VulkanSpaceToBatchND(const Op* op, Backend* bn);
    virtual ~VulkanSpaceToBatchND();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    int mPadTop;
    int mPadLeft;
    int mBlockShapeHeight;
    int mBlockShapeWidth;
    std::shared_ptr<VulkanBuffer> mGpuParam;
    const VulkanPipeline* mSpaceToBatchPipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDescriptorSet;
    const VulkanSampler* mSampler;
};

} // namespace MNN

#endif /* VulkanSpaceToBatchND_hpp */
