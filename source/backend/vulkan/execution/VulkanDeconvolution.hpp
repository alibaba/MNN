//
//  VulkanDeconvolution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanDeconvolution_hpp
#define VulkanDeconvolution_hpp
#include "VulkanBasicExecution.hpp"
#include "VulkanConvolution.hpp"
#include "VulkanMatrixMultier.hpp"
namespace MNN {
class VulkanDeconvolution : public VulkanBasicExecution {
public:
    virtual ~VulkanDeconvolution() {
    }

    VulkanDeconvolution(Backend* bn, const Convolution2D* conv);
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;

    static void writeConvolutionConst(VulkanConvolutionCommon::ConvolutionParameter* parameter,
                                      const Convolution2DCommon* common, const Tensor* src, const Tensor* dst);

private:
    std::shared_ptr<VulkanMatrixMultier> mMultiler;
    std::shared_ptr<VulkanImage> mBias;

    const VulkanPipeline* mIm2Col;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mIm2ColSet;

    const VulkanPipeline* mCol2Im;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mCol2ImSet;
    const VulkanSampler* mSampler;

    const Convolution2DCommon* mConvCommonOption;
    std::shared_ptr<VulkanBuffer> mConvParam;
};
} // namespace MNN
#endif /* VulkanDeconvolution_hpp */
