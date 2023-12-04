//
//  VulkanDeconvolutionDepthwise.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanDeconvolutionDepthwise_hpp
#define VulkanDeconvolutionDepthwise_hpp
#include "VulkanBasicExecution.hpp"
#include "VulkanDeconvolution.hpp"

namespace MNN {
class VulkanDeconvolutionDepthwise : public VulkanBasicExecution {
public:
    virtual ~VulkanDeconvolutionDepthwise() {
    }

    VulkanDeconvolutionDepthwise(Backend* bn, const Convolution2D* conv);
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanImage> mBias;
    std::shared_ptr<VulkanImage> mKernel;

    const VulkanPipeline* mPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mPipelineSet;

    const VulkanSampler* mSampler;

    const Convolution2DCommon* mConvCommonOption;
    std::shared_ptr<VulkanBuffer> mConvParam;

    int mLocalSize[3];
};
} // namespace MNN
#endif /* VulkanDeconvolutionDepthwise_hpp */
