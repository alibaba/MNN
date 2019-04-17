//
//  VulkanBatchToSpaceND.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanBatchToSpaceND_hpp
#define VulkanBatchToSpaceND_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanBatchToSpaceND : public VulkanBasicExecution {
public:
    VulkanBatchToSpaceND(const Op* op, Backend* bn);
    virtual ~VulkanBatchToSpaceND();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    int mCropTop;
    int mCropLeft;
    int mBlockShapeHeight;
    int mBlockShapeWidth;
    std::shared_ptr<VulkanBuffer> mGpuParam;
    const VulkanPipeline* mBatchToSpacePipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDescriptorSet;
    const VulkanSampler* mSampler;
};

} // namespace MNN
#endif /* VulkanBatchToSpaceND_hpp */
