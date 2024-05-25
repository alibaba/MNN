//
//  VulkanUnary.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanUnary_hpp
#define VulkanUnary_hpp

#include "VulkanBasicExecution.hpp"
#include "core/TensorUtils.hpp"
#include <array>
namespace MNN {

class VulkanUnary : public VulkanBasicExecution {
public:
    VulkanUnary(const std::string& midType, Backend* bn, bool image = false);
    virtual ~VulkanUnary();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;
    bool encode(const Tensor* input, const Tensor* output, const VulkanCommandPool::Buffer* cmdBuffer, const Tensor::InsideDescribe::Region* region);
    bool encoderSingle(const VulkanCommandPool::Buffer* cmdBuffer, const VulkanImage* dest, const VulkanImage* source,
                       const std::array<int, 3>& size
                       );

private:
    std::vector<std::shared_ptr<VulkanBuffer>> mParams;
    const VulkanPipeline* mUnaryPipeline;
    std::vector<std::shared_ptr<VulkanLayout::DescriptorSet>> mDesSet;
};

} // namespace MNN

#endif /* VulkanUnary_hpp */
