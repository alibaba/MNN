//
//  VulkanDeconvolutionDepthwise.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanDeconvolutionDepthwise.hpp"
#include "Macro.h"
namespace MNN {
VulkanDeconvolutionDepthwise::VulkanDeconvolutionDepthwise(Backend* bn, const Convolution2D* conv)
    : VulkanBasicExecution(bn) {
    mConvCommonOption = conv->common();
    auto vkBn         = (VulkanBackend*)bn;
    int outputC4      = UP_DIV(mConvCommonOption->outputCount(), 4);
    mBias             = std::make_shared<VulkanImage>(vkBn->getMemoryPool(), false, std::vector<int>{outputC4, 1});
    {
        auto biasBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, outputC4 * 4 * sizeof(float));
        auto biasPtr    = biasBuffer->map();
        ::memset(biasPtr, 0, outputC4 * 4 * sizeof(float));
        ::memcpy(biasPtr, conv->bias()->data(), conv->bias()->size() * sizeof(float));
        biasBuffer->unmap();
        vkBn->copyBufferToImage(biasBuffer.get(), mBias.get());
    }
    mConvParam = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false,
                                                sizeof(VulkanConvolutionCommon::ConvolutionParameter), nullptr,
                                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    int kh     = mConvCommonOption->kernelY();
    int kw     = mConvCommonOption->kernelX();
    int co     = mConvCommonOption->outputCount();
    int coC4   = UP_DIV(co, 4);
    mKernel    = std::make_shared<VulkanImage>(vkBn->getMemoryPool(), false, std::vector<int>{kw * kh, coC4});

    const int alignedWeightSize = kh * kw * ALIGN_UP4(co);
    auto tempWeightBuffer =
        std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, alignedWeightSize * sizeof(float));
    auto tempReorderWeight = (float*)tempWeightBuffer->map();
    ::memset(tempReorderWeight, 0, alignedWeightSize * sizeof(float));
    auto tempWeight = conv->weight()->data();
    for (int b = 0; b < co; ++b) {
        int b_4      = b / 4;
        float* dst_b = tempReorderWeight + b_4 * 4 * kw * kh;
        int mx       = b % 4;
        for (int y = 0; y < kh; ++y) {
            float* dst_y = dst_b + y * kw * 4;
            for (int x = 0; x < kw; ++x) {
                float* dst_x = dst_y + x * 4;
                dst_x[mx]    = tempWeight[x + y * kw + b * kw * kh];
            }
        }
    }
    tempWeightBuffer->unmap();

    vkBn->copyBufferToImage(tempWeightBuffer.get(), mKernel.get());
    mSampler = vkBn->getCommonSampler();

    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    std::string macro = VulkanConvolutionCommon::getPostTreatMacro(mConvCommonOption);

    mPipeline = vkBn->getPipeline("glsl_deconvolutionDepthwise_" + macro + "comp", types);
    mPipelineSet.reset(mPipeline->createSet());

    mLocalSize[0] = 8;
    mLocalSize[1] = 8;
    mLocalSize[2] = 1;
}

ErrorCode VulkanDeconvolutionDepthwise::onEncode(const std::vector<Tensor*>& inputs,
                                                 const std::vector<Tensor*>& outputs,
                                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    auto src         = inputs[0];
    auto dst         = outputs[0];
    const int ocDiv4 = UP_DIV(dst->channel(), 4);
    auto common      = mConvCommonOption;
    {
        auto convCons = reinterpret_cast<VulkanConvolutionCommon::ConvolutionParameter*>(mConvParam->map());
        VulkanDeconvolution::writeConvolutionConst(convCons, common, src, dst);
        mConvParam->unmap();
    }
    mPipelineSet->writeImage((VkImageView)dst->deviceId(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
    mPipelineSet->writeImage((VkImageView)src->deviceId(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 1);
    mPipelineSet->writeImage(mKernel->view(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 2);
    mPipelineSet->writeImage(mBias->view(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 3);
    mPipelineSet->writeBuffer(mConvParam->buffer(), 4, mConvParam->size());
    mPipeline->bind(cmdBuffer->get(), mPipelineSet->get());

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(dst->width(), mLocalSize[0]), UP_DIV(dst->height(), mLocalSize[1]),
                  UP_DIV(ocDiv4, mLocalSize[2]));

    return NO_ERROR;
}

class VulkanDeconvolutionDepthwiseCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new VulkanDeconvolutionDepthwise(backend, op->main_as_Convolution2D());
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_DeconvolutionDepthwise, new VulkanDeconvolutionDepthwiseCreator);
    return true;
}();
} // namespace MNN
