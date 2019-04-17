//
//  VulkanSigmoid.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_VULKANSIGMOID_H
#define MNN_VULKANSIGMOID_H
#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanSigmoid : public VulkanBasicExecution {
public:
    VulkanSigmoid(const Op* op, Backend* bn);
    virtual ~VulkanSigmoid();

    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    const VulkanPipeline* mBufferPipeline;
    const VulkanPipeline* mImagePipeline;

protected:
    std::shared_ptr<VulkanBuffer> mArgs;
    std::shared_ptr<VulkanBuffer> mInputBuffer;
    std::shared_ptr<VulkanBuffer> mOutputBuffer;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDescriptorSet;
};
} // namespace MNN

#endif // MNN_VULKANSIGMOID_H
