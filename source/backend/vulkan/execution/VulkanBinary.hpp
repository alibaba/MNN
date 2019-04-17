//
//  VulkanBinary.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanBinary_hpp
#define VulkanBinary_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"

namespace MNN {
class VulkanBinary : public VulkanBasicExecution {
public:
    VulkanBinary(const Op*, Backend* bn);
    virtual ~VulkanBinary();

    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    const VulkanPipeline* mBinaryPipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDescriptorSet;
    const VulkanBackend* mVkBackend;
    int32_t mType;
};
} // namespace MNN

#endif /* VulkanBinary_hpp */
