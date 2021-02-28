//
//  VulkanConvolutionImpl.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanConvolutionImpl.hpp"
#include "core/Macro.h"
#include "VulkanConvolution.hpp"
#include "VulkanConvolutionWinograd.hpp"
#include "VulkanMatMul.hpp"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
namespace MNN {
//#define VULKAN_IM2COL_GEMM_UNIT 512
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
            backend->copyBufferToImage(tempBias.get(), mBias.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        mConvCons = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false,
                                                   sizeof(VulkanConvolutionCommon::ConvolutionParameter), nullptr,
                                                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        {
            auto reorderWeight =
                VulkanConvolutionImpl::createBufferForSlideWindow(extra, convOption, weightPtr, ci, co);
            mKernel = std::make_shared<VulkanImage>(extra->getMemoryPool(), false,
                                                    std::vector<int>{ALIGN_UP4(ci), UP_DIV(co, 4), kh * kw});
            extra->copyBufferToImage(reorderWeight.get(), mKernel.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
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
            mConvSet->writeImage(((VulkanTensor*)output->deviceId())->image()->view(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
            mConvSet->writeImage(((VulkanTensor*)input->deviceId())->image()->view(), mSampler->get(),
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

static void writeParameters(VulkanMatMul::Reorder::nchwBuffer& parameters, int co, int ci, int kh, int kw) {
    parameters.size[0] = co;
    parameters.size[1] = ci;
    parameters.size[2] = kh;
    parameters.size[3] = kw;
    parameters.stride[0] = ci * kh * kw;
    parameters.stride[1] = kh * kw;
    parameters.stride[2] = kw;
    parameters.stride[3] = 1;
}
class VulkanConvolutionIm2Col : public VulkanBasicExecution {
public:

    VulkanConvolutionIm2Col(VulkanBackend* backend, const Convolution2DCommon* convOption, const float* weightPtr,
                            const float* biasPtr, int ci, int co) : VulkanBasicExecution(backend), mConvCommonOption(convOption) {
        auto kw = convOption->kernelX();
        auto kh = convOption->kernelY();
        if (nullptr != weightPtr) {
            // Static weight
            VulkanMatMul::Reorder reorder(backend, true);
            VulkanMatMul::Reorder::nchwBuffer parameters;
            writeParameters(parameters, co, ci, kh, kw);
            mKernel = VulkanMatrixMultier4x4::createKernel(backend, nullptr, ALIGN_UP4(ci) * kh * kw, co, 1);
            auto weightSize = ci * co * kh * kw;
            std::shared_ptr<VulkanBuffer> tempBuffer(new VulkanBuffer(backend->getMemoryPool(), false, weightSize*sizeof(float), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
            auto tempWeightBuffer = tempBuffer->map();
            ::memcpy(tempWeightBuffer, weightPtr, weightSize * sizeof(float));
            tempBuffer->unmap();
            std::shared_ptr<VulkanBuffer> tempBuffer2(new VulkanBuffer(backend->getMemoryPool(), false, reorder.computeMiddleBufferSize(co, kh, kw, ci) *sizeof(float), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
            std::shared_ptr<VulkanCommandPool::Buffer> cmdBuffer(backend->getPool().allocBuffer());
            cmdBuffer->begin(0);
            reorder.encode(tempBuffer->buffer(), tempBuffer->size(), tempBuffer2->buffer()
                           , tempBuffer2->size(), mKernel.get(), cmdBuffer.get(), parameters);
            cmdBuffer->end();
            backend->getPool().submitAndWait(cmdBuffer->get());
        }
        mMultiCreator = [ci, kh, kw, co, backend, this]() {
            auto multi = std::make_shared<VulkanMatrixMultier4x4>(backend, nullptr, ALIGN_UP4(ci) * kh * kw, co, 1, mKernel);
            return multi;
        };
        std::vector<VkDescriptorType> im2Coltypes{
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        if (kw == 1 && kh == 1 && convOption->padX() == 0 && convOption->padY() == 0) {
            mIm2Col =
                backend->getPipeline("glsl_im2col1x1_comp", /* glsl_im2col1x1_comp, glsl_im2col1x1_comp_len,*/ im2Coltypes);
        } else {
            mIm2Col = backend->getPipeline("glsl_im2col_comp", /*glsl_im2col_comp, glsl_im2col_comp_len,*/ im2Coltypes);
        }
        std::vector<VkDescriptorType> Col2imTypes{
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        auto macro = VulkanConvolutionCommon::getPostTreatMacro(convOption);
        mCol2Im    = backend->getPipeline("glsl_col2Im_" + macro + "comp", Col2imTypes);

        mSampler      = backend->getCommonSampler();
        if (nullptr != biasPtr) {
            // Static bias
            mBias         = std::make_shared<VulkanImage>(backend->getMemoryPool(), false, UP_DIV(co, 4), 1);
            auto tempBias = std::make_shared<VulkanBuffer>(backend->getMemoryPool(), false, sizeof(float) * ALIGN_UP4(co));
            auto bias     = tempBias->map();
            ::memset(bias, 0, sizeof(float) * ALIGN_UP4(co));
            ::memcpy(bias, biasPtr, sizeof(float) * co);
            tempBias->unmap();
            backend->copyBufferToImage(tempBias.get(), mBias.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }
    ~VulkanConvolutionIm2Col() {
        // Do nothing
    }
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override {
        auto src         = inputs[0];
        auto dst         = outputs[0];
        const int icDiv4 = UP_DIV(src->channel(), 4);
        const int ocDiv4 = UP_DIV(dst->channel(), 4);
        auto vkBn = (VulkanBackend*)backend();
        int limit = vkBn->proty().limits.maxImageDimension2D * 4;
#ifdef VULKAN_IM2COL_GEMM_UNIT
        limit = VULKAN_IM2COL_GEMM_UNIT;
#endif
        if (limit < dst->width()) {
            MNN_ERROR("Don't support width too large feature: %d x %d, limit = %d\n", dst->width(), dst->height(), limit);
            return NOT_SUPPORT;
        }
        int batchLoopNumber = 1;
        int heightLoopNumber = 1;
        int unitHeight = dst->height();
        int unitBatch = dst->batch();
        auto area = dst->width() * dst->height();
        if (limit < area) {
            batchLoopNumber = dst->batch();
            unitBatch = 1;
            unitHeight = limit / dst->width();
            heightLoopNumber = UP_DIV(dst->height(), unitHeight);
        } else if (limit < area * dst->batch()) {
            unitBatch = limit / area;
            batchLoopNumber = UP_DIV(dst->batch(), unitBatch);
        }
        int loopNumber = batchLoopNumber * heightLoopNumber;
        mConvParams.resize(loopNumber);
        mMultilers.resize(loopNumber);
        mIm2ColSet.resize(loopNumber);
        mCol2ImSet.resize(loopNumber);

        for (int i=0; i<batchLoopNumber; ++i) {
            int batchOffset = i * unitBatch;
            int currentBatch = dst->batch() - batchOffset;
            if (currentBatch > unitBatch) {
                currentBatch = unitBatch;
            }
            for (int j=0; j<heightLoopNumber; ++j) {
                int heightOffset = j * unitHeight;
                int currentHeight = dst->height() - heightOffset;
                if (currentHeight > unitHeight) {
                    currentHeight = unitHeight;
                }
                auto index = i * heightLoopNumber + j;
                auto totalNumberInput = currentBatch * icDiv4 * dst->width() * currentHeight;
                auto totalNumberOutput = currentBatch * ocDiv4 * dst->width() * currentHeight;
                mConvParams[index] = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false,
                                                        sizeof(VulkanConvolutionCommon::ConvolutionParameter), nullptr,
                                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
                {
                    auto convCons = reinterpret_cast<VulkanConvolutionCommon::ConvolutionParameter*>(mConvParams[index]->map());
                    VulkanConvolutionCommon::writeParameter(convCons, mConvCommonOption, src, dst);
                    convCons->batch = batchOffset;
                    convCons->hOffset = heightOffset;
                    convCons->outputSize[3] = currentBatch;
                    convCons->outputSize[1] = currentHeight;
                    mConvParams[index]->unmap();
                }
                mIm2ColSet[index].reset(mIm2Col->createSet());
                mCol2ImSet[index].reset(mCol2Im->createSet());
                mMultilers[index] = mMultiCreator();
                mMultilers[index]->prepare(cmdBuffer, dst->width() * currentHeight * currentBatch);
                auto mMultiler = mMultilers[index].get();
                if (true) {
                    auto colImage = mMultiler->source();
                    cmdBuffer->barrierImageIfNeeded(colImage, VK_IMAGE_LAYOUT_GENERAL);
                    mIm2ColSet[index]->writeImage(colImage->view(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
                    mIm2ColSet[index]->writeImage((reinterpret_cast<VulkanTensor*>(src->deviceId()))->image()->view(), mSampler->get(),
                                        VK_IMAGE_LAYOUT_GENERAL, 1);
                    mIm2ColSet[index]->writeBuffer(mConvParams[index]->buffer(), 2, mConvParams[index]->size());
                    mIm2Col->bind(cmdBuffer->get(), mIm2ColSet[index]->get());
                    vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalNumberInput, VulkanConvolutionCommon::gImage2ColLocal),
                                1, 1);
                    cmdBuffer->barrierImageIfNeeded(colImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                }
                mMultilers[index]->compute(cmdBuffer);
                if (true) {
                    auto dstImage = mMultiler->dest();
                    mCol2ImSet[index]->writeImage(dstImage->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0);
                    mCol2ImSet[index]->writeImage((reinterpret_cast<VulkanTensor*>(dst->deviceId()))->image()->view(), mSampler->get(),
                                        VK_IMAGE_LAYOUT_GENERAL, 1);

                    mCol2ImSet[index]->writeImage(mBias->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
                    mCol2ImSet[index]->writeBuffer(mConvParams[index]->buffer(), 3, mConvParams[index]->size());
                    mCol2Im->bind(cmdBuffer->get(), mCol2ImSet[index]->get());
                    cmdBuffer->barrierImageIfNeeded(dstImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                    // cmdBuffer->barrierImage(dstImage->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                    vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalNumberOutput, VulkanConvolutionCommon::gImage2ColLocal),
                                1, 1);
                }
            }
        }
        return NO_ERROR;
    }
private:
    const VulkanPipeline* mIm2Col;
    const VulkanPipeline* mCol2Im;
    const VulkanSampler* mSampler;

    std::shared_ptr<VulkanImage> mBias;
    std::shared_ptr<VulkanImage> mKernel;
    const Convolution2DCommon* mConvCommonOption;
    std::vector<std::shared_ptr<VulkanPipeline::DescriptorSet>> mCol2ImSet;
    std::vector<std::shared_ptr<VulkanPipeline::DescriptorSet>> mIm2ColSet;
    std::vector<std::shared_ptr<VulkanBuffer>> mConvParams;
    std::vector<std::shared_ptr<VulkanMatrixMultier4x4>> mMultilers;
    std::function<std::shared_ptr<VulkanMatrixMultier4x4>()> mMultiCreator;
};

VulkanBasicExecution* VulkanConvolutionImpl::create(VulkanBackend* backend, const Convolution2DCommon* convOption,
                                                         const std::vector<Tensor*>& inputs, const Tensor* output,
                                                         const float* weightPtr, const float* biasPtr, int ci, int co) {
    AUTOTIME;
    if (inputs.size() > 1) {
        return new VulkanConvolutionIm2Col(backend, convOption, weightPtr, biasPtr, ci, co);
    }
    auto imageLimit = backend->proty().limits.maxImageDimension1D;
    if (ALIGN_UP4(ci) * convOption->kernelX() * convOption->kernelY() > imageLimit) {
        return new VulkanConvolutionSlideWindow(backend, convOption, weightPtr, biasPtr, ci, co);
    }
    if (VulkanConvolutionWinograd::support(convOption)) {
        if (output->width() >= 4 && output->height() >= 4 && output->batch() == 1) {
            return new VulkanConvolutionWinograd(backend, convOption, weightPtr, biasPtr, ci, co);
        }
    }
    if (UP_DIV(output->width() * output->height(), 4) > imageLimit) {
        return new VulkanConvolutionSlideWindow(backend, convOption, weightPtr, biasPtr, ci, co);
    }
    return new VulkanConvolutionIm2Col(backend, convOption, weightPtr, biasPtr, ci, co);
}

} // namespace MNN
