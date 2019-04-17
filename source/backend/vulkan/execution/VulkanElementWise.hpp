//
//  VulkanElementWise.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanElementWise_hpp
#define VulkanElementWise_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"

namespace MNN {
class VulkanElementWise : public Execution {
public:
    VulkanElementWise(EltwiseType type, Backend *bn);
    virtual ~VulkanElementWise();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    const VulkanPipeline *mElemenWisePipeline;
    std::vector<std::shared_ptr<VulkanPipeline::DescriptorSet>> mSubDescriptorSets;
    std::vector<std::shared_ptr<VulkanCommandPool::Buffer>> mBuffers;
};
} // namespace MNN

#endif /* VulkanElementWise_hpp */
