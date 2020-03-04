//
//  VulkanConvolutionImpl.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/execution/VulkanConvolutionImpl.hpp"
#include "core/Macro.h"
#include "backend/vulkan/execution/VulkanConvolution.hpp"
#include "backend/vulkan/execution/VulkanConvolutionWinograd.hpp"
#include "backend/vulkan/execution/VulkanMatrixMultier.hpp"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
namespace MNN {

static int gPretreatLocalSize[] = {16, 16, 1};
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
    class BufferToImageCopy {
    public:
        BufferToImageCopy(const VulkanBackend* bn) {
            mBackend = bn;
            std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
            mPipeline = mBackend->getPipeline("glsl_buffer2Image2D_comp", types);
            mSets.reset(mPipeline->createSet());
            mConstBuffer = std::make_shared<VulkanBuffer>(bn->getMemoryPool(), true, 2 * sizeof(int),
                                                              nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        }
        void encode(const VulkanImage* image, VkBuffer buffer, size_t bufferSize, const VulkanCommandPool::Buffer* cmdBuffer) {
            int localX = 16;
            int localY = 16;
            int localZ = 1;
            int* dim = (int*)mConstBuffer->map();
            dim[0] = image->width();
            dim[1] = image->height();
            mConstBuffer->unmap();
            mSets->writeImage(image->view(), mBackend->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
            mSets->writeBuffer(buffer, 1, bufferSize);
            mSets->writeBuffer(mConstBuffer->buffer(), 2, mConstBuffer->size());
            mPipeline->bind(cmdBuffer->get(), mSets->get());
            cmdBuffer->barrierSource(buffer, 0, bufferSize);
            vkCmdDispatch(cmdBuffer->get(), UP_DIV(image->width(), localX), UP_DIV(image->height(), localY),
                          UP_DIV(image->depth(), localZ));
        }
    private:
        const VulkanBackend* mBackend;
        const VulkanPipeline* mPipeline;
        std::shared_ptr<VulkanPipeline::DescriptorSet> mSets;
        std::shared_ptr<VulkanBuffer> mConstBuffer;
    };
    class WeightReorder {
    public:
        struct nchwBuffer {
            int width;
            int height;
            int channel;
            int batch;
        };
        WeightReorder(const VulkanBackend* bn) {
            std::vector<VkDescriptorType> types{
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            };
            mFirst = bn->getPipeline("glsl_nchwTonc4hw4_comp", types);
            mFirstSet.reset(mFirst->createSet());
            mBackend = bn;
            mUnitBuffer.reset(new VulkanBuffer(bn->getMemoryPool(), true, sizeof(nchwBuffer), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
            
            mSecond = bn->getPipeline("glsl_kernelReorder_comp", {
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            });
            mSecondSet.reset(mSecond->createSet());
        }
        ~ WeightReorder() {
            // Do nothing
        }
        void encode(VkBuffer source, size_t sourceSize, VkBuffer middleBuffer, size_t middelBufferSize, const VulkanImage* dest, const VulkanCommandPool::Buffer* cmdBuffer, int b, int h, int w, int c) {
            // First: nchw to nc4hw4
            auto ptr = (nchwBuffer*)mUnitBuffer->map();
            ptr->width = w;
            ptr->batch= b;
            ptr->height = h;
            ptr->channel = c;
            mUnitBuffer->unmap();
            auto cDiv4 = UP_DIV(c, 4);
            mFirstSet->writeBuffer(middleBuffer, 1, middelBufferSize);
            mFirstSet->writeBuffer(source, 0, sourceSize, 0);
            mFirstSet->writeBuffer(mUnitBuffer->buffer(), 2, mUnitBuffer->size());

            mFirst->bind(cmdBuffer->get(), mFirstSet->get());
            cmdBuffer->barrierSource(source, 0, sourceSize);
            vkCmdDispatch(cmdBuffer->get(), UP_DIV(w, 2), UP_DIV(h, 2), UP_DIV(cDiv4 * b, 32));
            
            // Second: nc4hw4 to image2d
            mSecondSet->writeImage(dest->view(), mBackend->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
            mSecondSet->writeBuffer(middleBuffer, 1, middelBufferSize);
            mSecondSet->writeBuffer(mUnitBuffer->buffer(), 2, mUnitBuffer->size());
            mSecond->bind(cmdBuffer->get(), mSecondSet->get());
            cmdBuffer->barrierSource(middleBuffer, 0, middelBufferSize);
            vkCmdDispatch(cmdBuffer->get(), UP_DIV(cDiv4 * w * h, 16), UP_DIV(UP_DIV(b, 4), 16), 1);
        }
    private:
        const VulkanPipeline* mFirst;
        const VulkanPipeline* mSecond;
        std::shared_ptr<VulkanPipeline::DescriptorSet> mFirstSet;
        std::shared_ptr<VulkanPipeline::DescriptorSet> mSecondSet;
        const VulkanBackend* mBackend;
        std::shared_ptr<VulkanBuffer> mUnitBuffer;
    };
    VulkanConvolutionIm2Col(VulkanBackend* backend, const Convolution2DCommon* convOption, const float* weightPtr,
                            const float* biasPtr, int ci, int co) : VulkanBasicExecution(backend), mConvCommonOption(convOption) {
        auto kw = convOption->kernelX();
        auto kh = convOption->kernelY();
        mKernel = VulkanMatrixMultier::createKernel(backend, nullptr, ALIGN_UP4(ci) * kh * kw, co, 1);
        if (nullptr != weightPtr) {
            auto weightSize = ci * co * kh * kw;
            std::shared_ptr<VulkanBuffer> tempBuffer(new VulkanBuffer(backend->getMemoryPool(), false, weightSize*sizeof(float), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
            auto tempWeightBuffer = tempBuffer->map();
            ::memcpy(tempWeightBuffer, weightPtr, weightSize * sizeof(float));
            tempBuffer->unmap();
            std::shared_ptr<VulkanBuffer> tempBuffer2(new VulkanBuffer(backend->getMemoryPool(), false, ALIGN_UP4(ci) * co * kh * kw *sizeof(float), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
            std::shared_ptr<VulkanCommandPool::Buffer> cmdBuffer(backend->getPool().allocBuffer());
            cmdBuffer->begin(0);
            WeightReorder reorder(backend);
            reorder.encode(tempBuffer->buffer(), tempBuffer->size(), tempBuffer2->buffer()
                           , tempBuffer2->size(), mKernel.get(), cmdBuffer.get(), co, kh, kw, ci);
            cmdBuffer->end();
            backend->getPool().submitAndWait(cmdBuffer->get());
        }
        mMultiCreator = [ci, kh, kw, co, backend, this]() {
            auto multi = std::make_shared<VulkanMatrixMultier>(backend, nullptr, ALIGN_UP4(ci) * kh * kw, co, 1, mKernel);
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
        mBias         = std::make_shared<VulkanImage>(backend->getMemoryPool(), false, UP_DIV(co, 4), 1);
        if (nullptr != biasPtr) {
            auto tempBias = std::make_shared<VulkanBuffer>(backend->getMemoryPool(), false, sizeof(float) * ALIGN_UP4(co));
            auto bias     = tempBias->map();
            ::memset(bias, 0, sizeof(float) * ALIGN_UP4(co));
            ::memcpy(bias, biasPtr, sizeof(float) * co);
            tempBias->unmap();
            backend->copyBufferToImage(tempBias.get(), mBias.get());
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
        if (inputs.size() > 1) {
            int ci = inputs[1]->length(1);
            int co = inputs[1]->length(0);
            int kh = inputs[1]->length(2);
            int kw = inputs[1]->length(3);
            mTempWeightBuffer.reset(new VulkanBuffer(vkBn->getDynamicMemoryPool(), false, ALIGN_UP4(ci) * co * kh * kw *sizeof(float), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
            mWeightReorder.reset(new WeightReorder(vkBn));
            mWeightReorder->encode((VkBuffer)inputs[1]->deviceId(), inputs[1]->size(), mTempWeightBuffer->buffer()
                           , mTempWeightBuffer->size(), mKernel.get(), cmdBuffer, co, kh, kw, ci);
        }
        if (inputs.size() > 2) {
            mBiasCopy.reset(new BufferToImageCopy(vkBn));
            mBiasCopy->encode(mBias.get(), (VkBuffer)(inputs[2]->deviceId()), inputs[2]->size(), cmdBuffer);
            cmdBuffer->barrierImage(mBias->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        int permitMaxBatch = (vkBn->proty().limits.maxImageDimension1D * 4) / (dst->width() * dst->height());
        if (permitMaxBatch < 1) {
            MNN_ERROR("Don't support too large feature: %d x %d\n", dst->width(), dst->height());
            return NOT_SUPPORT;
        }
        auto unitBatch = permitMaxBatch;
        if (unitBatch > dst->batch()) {
            unitBatch = dst->batch();
        }
        int loopNumber = (dst->batch() + unitBatch - 1) / unitBatch;
        mConvParams.resize(loopNumber);
        mMultilers.resize(loopNumber);
        mIm2ColSet.resize(loopNumber);
        mCol2ImSet.resize(loopNumber);

        for (int i=0; i<loopNumber; ++i) {
            int batchOffset = i * unitBatch;
            int currentBatch = dst->batch() - batchOffset;
            if (currentBatch > unitBatch) {
                currentBatch = unitBatch;
            }
            mConvParams[i] = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false,
                                                    sizeof(VulkanConvolutionCommon::ConvolutionParameter), nullptr,
                                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            {
                auto convCons = reinterpret_cast<VulkanConvolutionCommon::ConvolutionParameter*>(mConvParams[i]->map());
                VulkanConvolutionCommon::writeParameter(convCons, mConvCommonOption, src, dst);
                convCons->batch = batchOffset;
                mConvParams[i]->unmap();
            }
            mIm2ColSet[i].reset(mIm2Col->createSet());
            mCol2ImSet[i].reset(mCol2Im->createSet());
            mMultilers[i] = mMultiCreator();
            mMultilers[i]->prepare(dst->width() * dst->height() * currentBatch);
            auto mMultiler = mMultilers[i].get();
            if (true) {
                auto colImage = mMultiler->source();
                mIm2ColSet[i]->writeImage(colImage->view(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
                mIm2ColSet[i]->writeImage((reinterpret_cast<VkImageView>(src->deviceId())), mSampler->get(),
                                    VK_IMAGE_LAYOUT_GENERAL, 1);
                mIm2ColSet[i]->writeBuffer(mConvParams[i]->buffer(), 2, mConvParams[i]->size());
                mIm2Col->bind(cmdBuffer->get(), mIm2ColSet[i]->get());
                vkCmdDispatch(cmdBuffer->get(), UP_DIV(dst->width(), gPretreatLocalSize[0]),
                            UP_DIV(dst->height(), gPretreatLocalSize[1]), icDiv4 * currentBatch);
            }
            mMultilers[i]->compute(cmdBuffer);
            if (true) {
                auto dstImage = mMultiler->dest();
                mCol2ImSet[i]->writeImage(dstImage->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0);
                mCol2ImSet[i]->writeImage((reinterpret_cast<VkImageView>(dst->deviceId())), mSampler->get(),
                                    VK_IMAGE_LAYOUT_GENERAL, 1);

                mCol2ImSet[i]->writeImage(mBias->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
                mCol2ImSet[i]->writeBuffer(mConvParams[i]->buffer(), 3, mConvParams[i]->size());
                mCol2Im->bind(cmdBuffer->get(), mCol2ImSet[i]->get());
                cmdBuffer->barrierImage(dstImage->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                vkCmdDispatch(cmdBuffer->get(), UP_DIV(dst->width(), gPretreatLocalSize[0]),
                            UP_DIV(dst->height(), gPretreatLocalSize[1]), ocDiv4 * currentBatch);
            }
        }
        return NO_ERROR;
    }
private:
    std::shared_ptr<WeightReorder> mWeightReorder;
    std::shared_ptr<BufferToImageCopy> mBiasCopy;
    std::shared_ptr<VulkanBuffer> mTempWeightBuffer;

    const VulkanPipeline* mIm2Col;

    const VulkanPipeline* mCol2Im;
    const VulkanSampler* mSampler;

    std::shared_ptr<VulkanImage> mBias;
    std::shared_ptr<VulkanImage> mKernel;
    const Convolution2DCommon* mConvCommonOption;
    std::vector<std::shared_ptr<VulkanPipeline::DescriptorSet>> mCol2ImSet;
    std::vector<std::shared_ptr<VulkanPipeline::DescriptorSet>> mIm2ColSet;
    std::vector<std::shared_ptr<VulkanBuffer>> mConvParams;
    std::vector<std::shared_ptr<VulkanMatrixMultier>> mMultilers;
    std::function<std::shared_ptr<VulkanMatrixMultier>()> mMultiCreator;
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
        if (output->width() >= 4 && output->height() >= 4) {
            return new VulkanConvolutionWinograd(backend, convOption, weightPtr, biasPtr, ci, co);
        }
    }
    if (UP_DIV(output->width() * output->height(), 4) > imageLimit) {
        return new VulkanConvolutionSlideWindow(backend, convOption, weightPtr, biasPtr, ci, co);
    }
#ifdef MALI_SLIDE_OPT
    auto input = inputs[0];
    if (backend->gpuType() == VulkanBackend::MALI
        && (input->width() < gPretreatLocalSize[0] || input->height() < gPretreatLocalSize[1])
        //For mobilenet, use im2col
        && (input->channel() < 256 || output->channel() < 256)
        ) {
        return
        new VulkanConvolutionSlideWindow(backend, convOption, weightPtr,
                                         biasPtr, ci, co);
    }
#endif
    return new VulkanConvolutionIm2Col(backend, convOption, weightPtr, biasPtr, ci, co);
}

} // namespace MNN
