//
//  VulkanROIPooling.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanROIPooling_hpp
#define VulkanROIPooling_hpp
#include "VulkanBasicExecution.hpp"

namespace MNN {
class VulkanROIPooling : public VulkanBasicExecution {
public:
    VulkanROIPooling(Backend* bn, const float SpatialScale);
    virtual ~VulkanROIPooling();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    float mSpatialScale;
    std::shared_ptr<VulkanBuffer> mParamBuffer;
    const VulkanPipeline* mVulkanROIPoolingPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    const VulkanSampler* mSampler;
};
} // namespace MNN
#endif
