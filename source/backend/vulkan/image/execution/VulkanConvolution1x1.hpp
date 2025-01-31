//
//  VulkanConvolutionImpl.hpp
//  MNN
//
//  Created by MNN on 2025/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanConvolution1x1_hpp
#define VulkanConvolution1x1_hpp

#include "VulkanBasicExecution.hpp"


namespace MNN {

class VulkanConvolution1x1 : public VulkanBasicExecution {
public:
    VulkanConvolution1x1(VulkanBackend* backend, const Convolution2DCommon* convCommon, const float* weightPtr, const float* biasPtr, const int ic, const int oc);
    ~VulkanConvolution1x1();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    SharedPtr<VulkanPipeline> mPipeline;
    std::vector<SharedPtr<VulkanPipeline>> mCands;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    std::shared_ptr<VulkanBuffer> mConv1x1Param;
    const Convolution2DCommon* mConvCommon;
    std::shared_ptr<VulkanImage> mKernel;
    std::shared_ptr<VulkanBuffer> mKernelBuffer;
    std::shared_ptr<VulkanImage> mBias;
    std::vector<std::vector<int>> mCandidataGws;
};

} // end namespace MNN

#endif /* VulkanConvolution1x1_hpp */
