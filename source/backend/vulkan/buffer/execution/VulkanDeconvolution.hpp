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
#include "VulkanRaster.hpp"
namespace MNN {
class VulkanDeconvolution : public VulkanBasicExecution {
public:
    virtual ~VulkanDeconvolution() {
    }

    static VulkanDeconvolution* create(Backend* bn, const Convolution2D* conv, OpType type, bool multiInputs);
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    VulkanDeconvolution(Backend* bn);
    std::shared_ptr<VulkanBuffer> mBias;
    std::shared_ptr<Tensor> mKernel;

    const VulkanPipeline* mPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mPipelineSet;

    const Convolution2DCommon* mConvCommonOption;
    std::shared_ptr<VulkanBuffer> mConvParam;
    VulkanRaster::Componet mKernelReorder;
};
} // namespace MNN
#endif /* VulkanDeconvolution_hpp */
