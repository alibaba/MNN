//
//  VulkanPRelu.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanPRelu_hpp
#define VulkanPRelu_hpp

#include <stdio.h>

#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanPrelu : public VulkanBasicExecution {
public:
    VulkanPrelu(Backend* bn, const Op* op, Tensor* tensor);
    virtual ~VulkanPrelu();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;
    bool onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) override;

private:
    VulkanPrelu(Backend* bn, const VulkanPrelu* src);
    std::shared_ptr<VulkanBuffer> mGpuPreluParam;
    std::shared_ptr<VulkanBuffer> mSlope;
    const VulkanPipeline* mPreluPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
};

} // namespace MNN

#endif /* VulkanRelu_hpp */
