//
//  VulkanDeconvolution.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanDeconvolution.hpp"
#include "Macro.h"
namespace MNN {
VulkanDeconvolution::VulkanDeconvolution(Backend* bn, const Convolution2D* conv) : VulkanBasicExecution(bn) {
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
    int ci     = conv->weight()->size() / kh / kw / co;
    int ciC4   = UP_DIV(ci, 4);

    const int alignedWeightSize = ALIGN_UP4(ci) * kh * kw * ALIGN_UP4(co);
    // std::make_unique need c++14
    // std::shared_ptr does not support array
    std::unique_ptr<float[]> tempReorderWeight(new float[alignedWeightSize]);
    ::memset(tempReorderWeight.get(), 0, alignedWeightSize * sizeof(float));

    auto tempWeight = conv->weight()->data();
    for (int b = 0; b < co; ++b) {
        int b_4      = b / 4;
        float* dst_b = tempReorderWeight.get() + b_4 * 16 * kw * kh * ciC4;
        int mx       = b % 4;
        for (int d = 0; d < ci; ++d) {
            int my       = d % 4;
            int d_4      = d / 4;
            float* dst_d = dst_b + d_4 * 16;
            for (int y = 0; y < kh; ++y) {
                float* dst_y = dst_d + y * kw * 16 * ciC4;
                for (int x = 0; x < kw; ++x) {
                    float* dst_x       = dst_y + x * 16 * ciC4;
                    dst_x[4 * my + mx] = tempWeight[x + y * kw + b * kw * kh + d * kw * kh * co];
                }
            }
        }
    }
    mMultiler =
        std::make_shared<VulkanMatrixMultier>(vkBn, tempReorderWeight.get(), ALIGN_UP4(ci), ALIGN_UP4(co) * kh * kw);

    {
        std::vector<VkDescriptorType> im2ColTypes{
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        auto macro = VulkanConvolutionCommon::getPostTreatMacro(mConvCommonOption);
        mIm2Col    = vkBn->getPipeline("glsl_deconvIm2Col_" + macro + "comp", im2ColTypes);
        mIm2ColSet.reset(mIm2Col->createSet());
    }
    {
        std::vector<VkDescriptorType> col2ImTypes{
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        mCol2Im = vkBn->getPipeline("glsl_deconvCol2Im_comp", col2ImTypes);
        mCol2ImSet.reset(mCol2Im->createSet());
    }
    mSampler = vkBn->getCommonSampler();
}

void VulkanDeconvolution::writeConvolutionConst(VulkanConvolutionCommon::ConvolutionParameter* convCons,
                                                const Convolution2DCommon* common, const Tensor* src,
                                                const Tensor* dst) {
    const int icDiv4 = UP_DIV(src->channel(), 4);
    const int ocDiv4 = UP_DIV(dst->channel(), 4);
    int padX         = common->padX();
    int padY         = common->padY();

    if (common->padMode() == PadMode_SAME) {
        int output_width  = dst->width();
        int output_height = dst->height();

        int output_width_padded  = (src->width() - 1) * common->strideX() + common->kernelX();
        int output_height_padded = (src->height() - 1) * common->strideY() + common->kernelY();

        int pad_needed_width  = output_width_padded - output_width;
        int pad_needed_height = output_height_padded - output_height;

        padX = pad_needed_width / 2;
        padY = pad_needed_height / 2;
    }
    convCons->batch         = src->batch();
    convCons->dilate[0]     = common->dilateX();
    convCons->dilate[1]     = common->dilateY();
    convCons->stride[0]     = common->strideX();
    convCons->stride[1]     = common->strideY();
    convCons->pad[0]        = padX;
    convCons->pad[1]        = padY;
    convCons->kernelSize[0] = common->kernelX();
    convCons->kernelSize[1] = common->kernelY();

    convCons->inputSize[0] = src->width();
    convCons->inputSize[1] = src->height();
    convCons->inputSize[2] = icDiv4;
    convCons->inputSize[3] = src->batch();

    convCons->outputSize[0] = dst->width();
    convCons->outputSize[1] = dst->height();
    convCons->outputSize[2] = ocDiv4;
    convCons->outputSize[3] = dst->batch();
    convCons->group         = convCons->group;
}

ErrorCode VulkanDeconvolution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                        const VulkanCommandPool::Buffer* cmdBuffer) {
    auto src         = inputs[0];
    auto dst         = outputs[0];
    const int icDiv4 = UP_DIV(src->channel(), 4);
    const int ocDiv4 = UP_DIV(dst->channel(), 4);
    {
        auto convCons = reinterpret_cast<VulkanConvolutionCommon::ConvolutionParameter*>(mConvParam->map());
        writeConvolutionConst(convCons, mConvCommonOption, src, dst);
        mConvParam->unmap();
    }

    mMultiler->prepare(src->width() * src->height() * src->batch());
    if (true) {
        auto dstImage = mMultiler->source();
        mCol2ImSet->writeImage((reinterpret_cast<VkImageView>(src->deviceId())), mSampler->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0);
        mCol2ImSet->writeImage(dstImage->view(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 1);

        mCol2ImSet->writeBuffer(mConvParam->buffer(), 2, mConvParam->size());
        mCol2Im->bind(cmdBuffer->get(), mCol2ImSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(src->width(), 16), UP_DIV(src->height(), 16), icDiv4 * src->batch());
    }

    mMultiler->compute(cmdBuffer);
    if (true) {
        auto dstImage = mMultiler->dest();
        mIm2ColSet->writeImage((reinterpret_cast<VkImageView>(dst->deviceId())), mSampler->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 0);
        mIm2ColSet->writeImage(dstImage->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        mIm2ColSet->writeImage(mBias->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
        mIm2ColSet->writeBuffer(mConvParam->buffer(), 3, mConvParam->size());
        mIm2Col->bind(cmdBuffer->get(), mIm2ColSet->get());
        cmdBuffer->barrierImage(dstImage->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(dst->width(), 16), UP_DIV(dst->height(), 16), ocDiv4 * dst->batch());
    }

    return NO_ERROR;
}
class VulkanDeconvolutionCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new VulkanDeconvolution(backend, op->main_as_Convolution2D());
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Deconvolution, new VulkanDeconvolutionCreator);
    return true;
}();
} // namespace MNN
