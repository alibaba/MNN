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
    VulkanBinary(const std::string& shaderName, Backend* bn, bool image, int number, int activationType = 0);
    virtual ~VulkanBinary();

    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::vector<std::shared_ptr<VulkanBuffer>> mConstBuffer;
    const VulkanPipeline* mBinaryPipeline;
    std::vector<std::shared_ptr<VulkanLayout::DescriptorSet>> mDescriptorSet;
    int mActivationType = 0;
};
} // namespace MNN

#endif /* VulkanBinary_hpp */
