//
//  VulkanResize.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanResize_hpp
#define VulkanResize_hpp
#include "VulkanBasicExecution.hpp"

namespace MNN {
class VulkanResize : public VulkanBasicExecution {
public:
    VulkanResize(Backend* bn, float xScale, float yScale, int resizeType=2);
    virtual ~VulkanResize();
    ErrorCode encodeImpl(Tensor* input, Tensor* output, const float* transform,
                         const VulkanCommandPool::Buffer* cmdBuffer);

private:
    float mXScale;
    float mYScale;
    std::shared_ptr<VulkanBuffer> mParamBuffer;
    const VulkanPipeline* mVulkanResizePipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
};
} // namespace MNN
#endif
