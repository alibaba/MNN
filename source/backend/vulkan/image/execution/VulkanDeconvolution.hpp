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
#include "VulkanMatMul.hpp"

namespace MNN {
class VulkanDeconvolution : public VulkanBasicExecution {
public:
    virtual ~VulkanDeconvolution() {
    }

    VulkanDeconvolution(Backend* bn, const std::vector<Tensor*>& inputs, const Convolution2D* conv);
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;

    static void writeConvolutionConst(VulkanConvolutionCommon::ConvolutionParameter* parameter,
                                      const Convolution2DCommon* common, const Tensor* src, const Tensor* dst);

private:
    std::shared_ptr<VulkanMatrixMultier4x4> mMultiler;
    std::shared_ptr<VulkanImage> mBias;
    std::shared_ptr<VulkanConvolutionCommon::BufferToImageCopy> mBiasCopy;
    std::shared_ptr<VulkanImage> mKernel;
    std::shared_ptr<VulkanMatMul::Reorder> mReorder;
    std::shared_ptr<VulkanBuffer> mMidBuffer;

    const VulkanPipeline* mIm2Col;
    std::shared_ptr<VulkanLayout::DescriptorSet> mIm2ColSet;

    const VulkanPipeline* mCol2Im;
    std::shared_ptr<VulkanLayout::DescriptorSet> mCol2ImSet;
    const VulkanSampler* mSampler;

    const Convolution2DCommon* mConvCommonOption;
    std::shared_ptr<VulkanBuffer> mConvParam;
};
} // namespace MNN
#endif /* VulkanDeconvolution_hpp */
