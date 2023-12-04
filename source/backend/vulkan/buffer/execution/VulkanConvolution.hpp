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
#include "core/ConvolutionCommon.hpp"
namespace MNN {
class VulkanConvolutionCommon : public VulkanBasicExecution {
public:
    VulkanConvolutionCommon(const Convolution2DCommon* common, Backend* bn);
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
        ivec4 offset;
    };

    static void writeParameter(ConvolutionParameter* dest, const Convolution2DCommon* common, const Tensor* input,
                               const Tensor* output);
    static void writeDeconvolution(ConvolutionParameter* convCons,
                                                    const Convolution2DCommon* common, const Tensor* src,
                                   const Tensor* dst);

    static std::string getPostTreatMacro(const Convolution2DCommon* common);
    static int gImage2ColLocal;
protected:
    virtual ErrorCode onEncodeConvolution(const Convolution2DCommon* common, const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs,
                                          const VulkanCommandPool::Buffer* cmdBuffer,
                                          const VulkanBuffer* constConvBuffer) = 0;

private:
    const Convolution2DCommon* mCommon;
    std::shared_ptr<VulkanBuffer> mConvCons;
};

class VulkanConvolutionDepthwise : public VulkanConvolutionCommon {
public:
    VulkanConvolutionDepthwise(const float* weightData, size_t weightSize, const Op* op, Backend* bn);
    virtual ~VulkanConvolutionDepthwise();
    virtual ErrorCode onEncodeConvolution(const Convolution2DCommon* common, const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs,
                                          const VulkanCommandPool::Buffer* cmdBuffer,
                                          const VulkanBuffer* constConvBuffer) override;
    virtual bool onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) override;

private:
    VulkanConvolutionDepthwise(const Op* op, Backend* bn);
    bool _init(const float* weightData, size_t weightSize, const Op* op, Backend* bn, bool initWeights);

    const VulkanPipeline* mConvPipeline;

    std::shared_ptr<VulkanLayout::DescriptorSet> mConvSet;
    std::shared_ptr<VulkanBuffer> mBias;
    std::shared_ptr<VulkanBuffer> mKernel;
    std::shared_ptr<VulkanLayout::DescriptorSet> mExtraSets;
    std::shared_ptr<VulkanBuffer> mExtraBuffers;

    int mLocalX = 0;
    int mLocalY = 0;
};
} // namespace MNN

#endif /* VulkanConvolution_hpp */
