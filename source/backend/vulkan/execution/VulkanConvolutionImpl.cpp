//
//  VulkanConvolutionImpl.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanConvolutionImpl.hpp"
#include "Macro.h"
#include "VulkanConvolution.hpp"
#include "VulkanConvolutionWinograd.hpp"
#include "VulkanMatrixMultier.hpp"
namespace MNN {
static int gPretreatLocalSize[3] = {16, 16, 1};
std::shared_ptr<VulkanBuffer> VulkanConvolutionImpl::createBufferForSlideWindow(const VulkanBackend* extra,
                                                                                const Convolution2DCommon* convOption,
                                                                                const float* weightPtr, int ci,
                                                                                int co) {
    int kw                      = convOption->kernelX();
    int kh                      = convOption->kernelY();
    const int alignedWeightSize = ALIGN_UP4(ci) * kh * kw * ALIGN_UP4(co);
    auto ciC4                   = UP_DIV(ci, 4);
    auto coC4                   = UP_DIV(co, 4);
    auto reorderWeight =
        std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false, alignedWeightSize * sizeof(float));
    auto destWeight = (float*)reorderWeight->map();
    ::memset(destWeight, 0, alignedWeightSize * sizeof(float));
    int kC = kw * kh;
    for (int oz = 0; oz < co; ++oz) {
        auto srcOz  = weightPtr + oz * ci * kC;
        auto destOz = destWeight + (oz / 4) * ciC4 * 16 + (oz % 4);
        for (int sz = 0; sz < ci; ++sz) {
            auto destSz = destOz + (sz / 4) * 16 + (sz % 4) * 4;
            auto srcSz  = srcOz + sz * kC;
            for (int k = 0; k < kC; ++k) {
                destSz[k * 16 * ciC4 * coC4] = srcSz[k];
            }
        }
    }

    reorderWeight->unmap();
    return reorderWeight;
}

class VulkanConvolutionSlideWindow : public VulkanBasicExecution {
public:
    VulkanConvolutionSlideWindow(VulkanBackend* backend, const Convolution2DCommon* convOption, const float* weightPtr,
                                 const float* biasPtr, int ci, int co)
        : VulkanBasicExecution(backend) {
        auto extra = static_cast<VulkanBackend*>(backend);
        mCommon    = convOption;
        mSampler   = backend->getCommonSampler();
        int kw     = convOption->kernelX();
        int kh     = convOption->kernelY();
        mBias      = std::make_shared<VulkanImage>(backend->getMemoryPool(), false, UP_DIV(co, 4), 1);
        {
            auto tempBias =
                std::make_shared<VulkanBuffer>(backend->getMemoryPool(), false, sizeof(float) * ALIGN_UP4(co));
            auto bias = tempBias->map();
            ::memset(bias, 0, sizeof(float) * ALIGN_UP4(co));
            ::memcpy(bias, biasPtr, sizeof(float) * co);
            tempBias->unmap();
            backend->copyBufferToImage(tempBias.get(), mBias.get());
        }

        mConvCons = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false,
                                                   sizeof(VulkanConvolutionCommon::ConvolutionParameter), nullptr,
                                                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        {
            auto reorderWeight =
                VulkanConvolutionImpl::createBufferForSlideWindow(extra, convOption, weightPtr, ci, co);
            mKernel = std::make_shared<VulkanImage>(extra->getMemoryPool(), false,
                                                    std::vector<int>{ALIGN_UP4(ci), UP_DIV(co, 4), kh * kw});
            extra->copyBufferToImage(reorderWeight.get(), mKernel.get());
        }
        // Create Pipeline
        std::vector<VkDescriptorType> convTypes{
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        auto common = mCommon;
        if (common->relu()) {
            mConvPipeline =
                extra->getPipeline("glsl_convolution_RELU_comp",
                                   /* glsl_convolution_RELU_comp, glsl_convolution_RELU_comp_len,*/ convTypes);
        } else if (common->relu6()) {
            mConvPipeline =
                extra->getPipeline("glsl_convolution_RELU6_comp",
                                   /* glsl_convolution_RELU6_comp, glsl_convolution_RELU6_comp_len,*/ convTypes);
        } else {
            mConvPipeline = extra->getPipeline("glsl_convolution_comp",
                                               /* glsl_convolution_comp, glsl_convolution_comp_len,*/ convTypes);
        }
        mLocalX = 2;
        mLocalY = 2;
        mLocalZ = 16;
    }
    ~VulkanConvolutionSlideWindow() {
    }
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override {
        auto input  = inputs[0];
        auto output = outputs[0];
        /*Set Const Parameters*/
        int ocDiv4    = UP_DIV(output->channel(), 4);
        int ow        = output->width();
        int oh        = output->height();
        auto convCons = reinterpret_cast<VulkanConvolutionCommon::ConvolutionParameter*>(mConvCons->map());
        VulkanConvolutionCommon::writeParameter(convCons, mCommon, input, output);
        mConvCons->unmap();

        /*Write Command Buffer*/
        if (true) {
            mConvSet.reset(mConvPipeline->createSet());
            mConvSet->writeImage((VkImageView)output->deviceId(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
            mConvSet->writeImage((VkImageView)input->deviceId(), mSampler->get(),
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
            mConvSet->writeImage(mKernel->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
            mConvSet->writeImage(mBias->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 3);
            mConvSet->writeBuffer(mConvCons->buffer(), 4, mConvCons->size());
            mConvPipeline->bind(cmdBuffer->get(), mConvSet->get());
            vkCmdDispatch(cmdBuffer->get(), UP_DIV(ow, mLocalX), UP_DIV(oh, mLocalY),
                          UP_DIV(ocDiv4 * input->batch(), mLocalZ));
        }
        return NO_ERROR;
    }

private:
    std::shared_ptr<VulkanImage> mBias;
    const Convolution2DCommon* mCommon;
    std::shared_ptr<VulkanBuffer> mConvCons;
    std::shared_ptr<VulkanImage> mKernel;
    const VulkanPipeline* mConvPipeline;

    std::shared_ptr<VulkanPipeline::DescriptorSet> mConvSet;
    const VulkanSampler* mSampler;

    int mLocalX = 0;
    int mLocalY = 0;
    int mLocalZ = 0;
};

class VulkanConvolutionIm2Col : public VulkanBasicExecution {
public:
    VulkanConvolutionIm2Col(VulkanBackend* backend, const Convolution2DCommon* convOption, const float* weightPtr,
                            const float* biasPtr, int ci, int co, int kh, int kw);
    ~VulkanConvolutionIm2Col();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;

public:
private:
    std::shared_ptr<VulkanMatrixMultier> mMultiler;

    const VulkanPipeline* mIm2Col;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mIm2ColSet;

    const VulkanPipeline* mCol2Im;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mCol2ImSet;
    const VulkanSampler* mSampler;

    std::shared_ptr<VulkanImage> mBias;
    const Convolution2DCommon* mConvCommonOption;
    std::shared_ptr<VulkanBuffer> mConvParam;
};

VulkanConvolutionIm2Col::VulkanConvolutionIm2Col(VulkanBackend* backend, const Convolution2DCommon* convOption,
                                                 const float* weightPtr, const float* biasPtr, int ci, int co, int kh,
                                                 int kw)
    : VulkanBasicExecution(backend), mConvCommonOption(convOption) {
    const int alignedWeightSize = ALIGN_UP4(ci) * kh * kw * ALIGN_UP4(co);
    // std::make_unique need c++14
    // std::shared_ptr does not support array
    std::unique_ptr<float[]> reorderedWeight(new float[alignedWeightSize]);
    ::memset(reorderedWeight.get(), 0, alignedWeightSize * sizeof(float));
    VulkanConvolutionImpl::MNNReorderWeight<float>(reorderedWeight.get(), weightPtr, ci, co, kh, kw);
    mMultiler = std::make_shared<VulkanMatrixMultier>(backend, reorderedWeight.get(), ALIGN_UP4(ci) * kh * kw, co);
    std::vector<VkDescriptorType> im2Coltypes{
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    if (kw == 1 && kh == 1 && convOption->padX() == 0 && convOption->padY() == 0) {
        mIm2Col =
            backend->getPipeline("glsl_im2col1x1_comp", /* glsl_im2col1x1_comp, glsl_im2col1x1_comp_len,*/ im2Coltypes);
    } else {
        mIm2Col = backend->getPipeline("glsl_im2col_comp", /*glsl_im2col_comp, glsl_im2col_comp_len,*/ im2Coltypes);
    }
    mIm2ColSet.reset(mIm2Col->createSet());

    std::vector<VkDescriptorType> Col2imTypes{
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    auto macro = VulkanConvolutionCommon::getPostTreatMacro(convOption);
    mCol2Im    = backend->getPipeline("glsl_col2Im_" + macro + "comp", Col2imTypes);
    mCol2ImSet.reset(mCol2Im->createSet());

    mSampler      = backend->getCommonSampler();
    mBias         = std::make_shared<VulkanImage>(backend->getMemoryPool(), false, UP_DIV(co, 4), 1);
    auto tempBias = std::make_shared<VulkanBuffer>(backend->getMemoryPool(), false, sizeof(float) * ALIGN_UP4(co));
    auto bias     = tempBias->map();
    ::memset(bias, 0, sizeof(float) * ALIGN_UP4(co));
    ::memcpy(bias, biasPtr, sizeof(float) * co);
    tempBias->unmap();
    backend->copyBufferToImage(tempBias.get(), mBias.get());

    mConvParam = std::make_shared<VulkanBuffer>(backend->getMemoryPool(), false,
                                                sizeof(VulkanConvolutionCommon::ConvolutionParameter), nullptr,
                                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
}

VulkanConvolutionIm2Col::~VulkanConvolutionIm2Col() {
}

template <typename T>
void VulkanConvolutionImpl::MNNReorderWeight(float* reorderedWeight, const T* srcWeight, int ci, int co, int kh, int kw,
                                             int unit) {
    const int alignedWeightSize = ALIGN_UP4(ci) * kh * kw * ALIGN_UP4(co);
    const int unit2             = unit * unit;
    int cur                     = 0;
    int batch_4                 = UP_DIV(co, unit);
    for (int b = 0; b < co; ++b) {
        int b_4  = b / unit;
        T* dst_b = reorderedWeight + b_4 * (alignedWeightSize / batch_4);
        int mx   = b % unit;
        for (int d = 0; d < ci; ++d) {
            int my   = d % unit;
            int d_4  = d / unit;
            T* dst_d = dst_b + d_4 * kw * kh * unit2;
            for (int y = 0; y < kh; ++y) {
                T* dst_y = dst_d + y * kw * unit2;
                for (int x = 0; x < kw; ++x) {
                    T* dst_x              = dst_y + x * unit2;
                    dst_x[unit * my + mx] = srcWeight[cur++];
                }
            }
        }
    }
}

ErrorCode VulkanConvolutionIm2Col::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                            const VulkanCommandPool::Buffer* cmdBuffer) {
    auto src         = inputs[0];
    auto dst         = outputs[0];
    const int icDiv4 = UP_DIV(src->channel(), 4);
    const int ocDiv4 = UP_DIV(dst->channel(), 4);
    {
        auto convCons = reinterpret_cast<VulkanConvolutionCommon::ConvolutionParameter*>(mConvParam->map());
        VulkanConvolutionCommon::writeParameter(convCons, mConvCommonOption, src, dst);
        mConvParam->unmap();
    }

    mMultiler->prepare(dst->width() * dst->height() * dst->batch());
    if (true) {
        auto colImage = mMultiler->source();
        mIm2ColSet->writeImage(colImage->view(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
        mIm2ColSet->writeImage((reinterpret_cast<VkImageView>(src->deviceId())), mSampler->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 1);
        mIm2ColSet->writeBuffer(mConvParam->buffer(), 2, mConvParam->size());
        mIm2Col->bind(cmdBuffer->get(), mIm2ColSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(dst->width(), gPretreatLocalSize[0]),
                      UP_DIV(dst->height(), gPretreatLocalSize[1]), icDiv4 * src->batch());
    }
    mMultiler->compute(cmdBuffer);
    if (true) {
        auto dstImage = mMultiler->dest();
        mCol2ImSet->writeImage(dstImage->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0);
        mCol2ImSet->writeImage((reinterpret_cast<VkImageView>(dst->deviceId())), mSampler->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 1);

        mCol2ImSet->writeImage(mBias->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
        mCol2ImSet->writeBuffer(mConvParam->buffer(), 3, mConvParam->size());
        mCol2Im->bind(cmdBuffer->get(), mCol2ImSet->get());
        cmdBuffer->barrierImage(dstImage->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(dst->width(), gPretreatLocalSize[0]),
                      UP_DIV(dst->height(), gPretreatLocalSize[1]), ocDiv4 * dst->batch());
    }

    return NO_ERROR;
}

std::shared_ptr<Execution> VulkanConvolutionImpl::create(VulkanBackend* backend, const Convolution2DCommon* convOption,
                                                         const Tensor* input, const Tensor* output,
                                                         const float* weightPtr, const float* biasPtr, int ci, int co) {
    auto imageLimit = backend->proty().limits.maxImageDimension1D;
    if (ALIGN_UP4(ci) * convOption->kernelX() * convOption->kernelY() > imageLimit) {
        return std::make_shared<VulkanConvolutionSlideWindow>(backend, convOption, weightPtr, biasPtr, ci, co);
    }

    if (VulkanConvolutionWinograd::support(convOption)) {
        if (output->width() >= 4 && output->height() >= 4) {
            return std::make_shared<VulkanConvolutionWinograd>(backend, convOption, weightPtr, biasPtr, ci, co);
        }
    }
    if (UP_DIV(output->width() * output->height(), 4) > imageLimit) {
        return std::make_shared<VulkanConvolutionSlideWindow>(backend, convOption, weightPtr, biasPtr, ci, co);
    }
    //    if (backend->gpuType() == VulkanBackend::MALI
    //        && (input->width() < gPretreatLocalSize[0] || input->height() < gPretreatLocalSize[1])
    //        //For mobilenet, use im2col
    //        && (input->channel() < 256 || output->channel() < 256)
    //        ) {
    //        return std::shared_ptr<Execution>(
    //                                          new VulkanConvolutionSlideWindow(backend, convOption, weightPtr,
    //                                          biasPtr, ci, co));
    //    }

    return std::make_shared<VulkanConvolutionIm2Col>(backend, convOption, weightPtr, biasPtr, ci, co,
                                                     convOption->kernelY(), convOption->kernelX());
}

} // namespace MNN
