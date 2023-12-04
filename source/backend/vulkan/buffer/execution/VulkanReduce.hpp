//
//  VulkanReduce.hpp
//  MNN
//
//  Created by MNN on 2020/03/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanReduce_hpp
#define VulkanReduce_hpp
#include "VulkanBasicExecution.hpp"
namespace MNN {
class VulkanReduce : public VulkanBasicExecution {
public:
    VulkanReduce(const std::string& name, const Op* op, Backend* bn);
    virtual ~VulkanReduce();

    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    const VulkanPipeline* mPipeline;
    const Op* mOp;
};

}


#endif
