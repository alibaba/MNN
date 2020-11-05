//
//  VulkanSpatialProduct.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanSpatialProduct_hpp
#define VulkanSpatialProduct_hpp
#include "VulkanBasicExecution.hpp"

namespace MNN {
class VulkanSpatialProduct : public VulkanBasicExecution {
public:
    VulkanSpatialProduct(const Op* op, Backend* bn);
    virtual ~VulkanSpatialProduct();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mParamBuffer;
    const VulkanPipeline* mVulkanSpatialProductPipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDescriptorSet;
    const VulkanSampler* mSampler;
};
} // namespace MNN
#endif
