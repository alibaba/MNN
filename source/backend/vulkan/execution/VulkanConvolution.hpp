//
//  VulkanConvolution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanConvolution_hpp
#define VulkanConvolution_hpp

#include "VulkanBasicExecution.hpp"
namespace MNN {
class VulkanConvolutionCommon : public VulkanBasicExecution {
public:
    VulkanConvolutionCommon(const Op* op, Backend* bn);
    virtual ~VulkanConvolutionCommon();

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;

    struct ConvolutionParameter {
        ivec2 pad;
        ivec2 kernelSize;
        ivec2 stride;
        ivec2 dilate;
        ivec4 inputSize;
        ivec4 outputSize;
        int batch;
        int group;
    };

    static void writeParameter(ConvolutionParameter* dest, const Convolution2DCommon* common, const Tensor* input,
                               const Tensor* output);
    static std::string getPostTreatMacro(const Convolution2DCommon* common);

protected:
    virtual ErrorCode onEncodeConvolution(const Convolution2DCommon* common, const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs,
                                          const VulkanCommandPool::Buffer* cmdBuffer,
                                          const VulkanBuffer* constConvBuffer, const VulkanImage* biasBuffer) = 0;

private:
    std::shared_ptr<VulkanImage> mBias;
    const Convolution2DCommon* mCommon;
    std::shared_ptr<VulkanBuffer> mConvCons;
};

class VulkanConvolutionDepthwise : public VulkanConvolutionCommon {
public:
    VulkanConvolutionDepthwise(const Op* op, Backend* bn);
    virtual ~VulkanConvolutionDepthwise();
    virtual ErrorCode onEncodeConvolution(const Convolution2DCommon* common, const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs,
                                          const VulkanCommandPool::Buffer* cmdBuffer,
                                          const VulkanBuffer* constConvBuffer, const VulkanImage* biasBuffer) override;

private:
    std::shared_ptr<VulkanImage> mKernel;

    const VulkanPipeline* mConvPipeline;

    std::shared_ptr<VulkanPipeline::DescriptorSet> mConvSet;
    const VulkanSampler* mSampler;

    int mLocalX = 0;
    int mLocalY = 0;
};
} // namespace MNN

#endif /* VulkanConvolution_hpp */
