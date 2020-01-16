//
//  VulkanConvolutionWinograd.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/execution/VulkanConvolutionWinograd.hpp"
#include <string.h>
#include "core/Macro.h"
#include "math/WingoradGenerater.hpp"
#define COMPUT_SIZE 4
#define COMPUT_SIZE2 16
#include "backend/vulkan/execution/VulkanConvolution.hpp"
namespace MNN {
struct WinogradConst {
    ivec4 inputSize;
    ivec4 outputSize;
    int padX;
    int padY;
    int unitWidth;
    int unitHeight;
    int unit;
};

bool VulkanConvolutionWinograd::support(const Convolution2DCommon* convOption) {
    if (convOption->strideX() != 1 || convOption->strideY() != 1) {
        return false;
    }
    if (convOption->dilateX() != 1 || convOption->dilateY() != 1) {
        return false;
    }
    if (convOption->kernelX() != convOption->kernelY()) {
        return false;
    }
    if (convOption->kernelX() != 3) {
        // [TODO] Support other kernel size
        return false;
    }
    if (convOption->kernelY() <= 1 || convOption->kernelY() >= COMPUT_SIZE) {
        return false;
    }
    if (convOption->group() != 1) {
        return false;
    }
    return true;
}

VulkanConvolutionWinograd::~VulkanConvolutionWinograd() {
}

VulkanConvolutionWinograd::VulkanConvolutionWinograd(VulkanBackend* backend, const Convolution2DCommon* convOption,
                                                     const float* weightPtr, const float* biasPtr, int ci, int co)
    : VulkanBasicExecution(backend) {
    MNN_ASSERT(support(convOption));
    mBackend = backend;
    mCommon  = convOption;
    mSampler = backend->getCommonSampler();
    mBias.reset(new VulkanImage(backend->getMemoryPool(), false, UP_DIV(co, 4), 1));
    {
        std::shared_ptr<VulkanBuffer> biasBuffer(
            new VulkanBuffer(backend->getMemoryPool(), false, ALIGN_UP4(co) * sizeof(float)));
        auto ptr = biasBuffer->map();
        ::memset(ptr, 0, ALIGN_UP4(co) * sizeof(float));
        ::memcpy(ptr, biasPtr, co * sizeof(float));
        biasBuffer->unmap();
        backend->copyBufferToImage(biasBuffer.get(), mBias.get());
    }
    int unit = COMPUT_SIZE - convOption->kernelY() + 1;
    mUnit    = unit;
    Math::WinogradGenerater generator(unit, convOption->kernelY(), 1.0f);

    mWinogradConst.reset(new VulkanBuffer(backend->getMemoryPool(), false, sizeof(WinogradConst), nullptr,
                                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    // Create Matrix Multier
    {
        auto ciC4 = UP_DIV(ci, 4);
        auto coC4 = UP_DIV(co, 4);
        std::shared_ptr<Tensor> originWeight(Tensor::create<float>(
            std::vector<int>{co, ci, (int)mCommon->kernelY(), (int)mCommon->kernelX()}, (void*)weightPtr, Tensor::CAFFE));
        auto weightDest = generator.allocTransformWeight(originWeight.get());
        generator.transformWeight(weightDest.get(), originWeight.get());
        mMultier.reset(new VulkanMatrixMultier(backend, weightDest->host<float>(), ciC4 * 4, coC4 * 4, COMPUT_SIZE2));
    }

    // Get transform pipeline
    {
        std::vector<VkDescriptorType> sourceTypes{
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        mSourceTransform = backend->getPipeline("glsl_winogradTransformSource2_3_1_comp", sourceTypes);
        std::vector<VkDescriptorType> destTypes{
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        auto macro     = VulkanConvolutionCommon::getPostTreatMacro(mCommon);
        mDestTransform = backend->getPipeline("glsl_winogradTransformDest2_3_1_" + macro + "comp", destTypes);
    }

    mTransformLocalSize[0] = 8;
    mTransformLocalSize[1] = 8;
    mTransformLocalSize[2] = 1;
}

ErrorCode VulkanConvolutionWinograd::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                              const VulkanCommandPool::Buffer* cmdBuffer) {
    auto src    = inputs[0];
    auto dst    = outputs[0];
    auto ow     = dst->width();
    auto oh     = dst->height();
    auto icC4   = UP_DIV(src->channel(), 4);
    auto ocC4   = UP_DIV(dst->channel(), 4);
    auto owUnit = UP_DIV(ow, mUnit);
    auto ohUnit = UP_DIV(oh, mUnit);
    int padX    = mCommon->padX();
    int padY    = mCommon->padY();
    if (mCommon->padMode() == PadMode_SAME) {
        int pad_needed_width  = (dst->width() - 1) * mCommon->strideX() + mCommon->kernelX() - src->width();
        int pad_needed_height = (dst->height() - 1) * mCommon->strideY() + mCommon->kernelY() - src->height();

        padX = pad_needed_width / 2;
        padY = pad_needed_height / 2;
    }
    int maxNumber      = (mBackend->proty().limits.maxImageDimension1D * 4) / COMPUT_SIZE2;
    int totalNumber    = owUnit * ohUnit;
    int sliceNumber    = 1;
    const int maxSlice = 100;
    if (maxNumber < totalNumber) {
        for (int i = 2; i < maxSlice; ++i) {
            int realNumber = UP_DIV(owUnit, i) * UP_DIV(ohUnit, i);
            if (realNumber < maxNumber) {
                sliceNumber = i;
                break;
            }
        }
    }
    int wPiece = UP_DIV(owUnit, sliceNumber);
    int hPiece = UP_DIV(ohUnit, sliceNumber);
    {
        auto value          = (WinogradConst*)mWinogradConst->map();
        value->inputSize[0] = src->width();
        value->inputSize[1] = src->height();
        value->inputSize[2] = icC4;
        value->inputSize[3] = src->batch();

        value->outputSize[0] = dst->width();
        value->outputSize[1] = dst->height();
        value->outputSize[2] = ocC4;
        value->outputSize[3] = dst->batch();

        value->padX       = padX;
        value->padY       = padY;
        value->unit       = mUnit;
        value->unitHeight = hPiece;
        value->unitWidth  = wPiece;
        mWinogradConst->unmap();
    }

    mMultier->prepare(wPiece * hPiece);
    mOffsetsBuffer.resize(sliceNumber * sliceNumber);
    mSourceTransformSet.resize(sliceNumber * sliceNumber);
    mDestTransformSet.resize(sliceNumber * sliceNumber);

    ivec2 offsetData;
    offsetData[0] = 0;
    offsetData[1] = 0;
    for (int y = 0; y < sliceNumber; ++y) {
        int hCount = hPiece;
        if (y == sliceNumber - 1) {
            hCount = ohUnit - (sliceNumber - 1) * hPiece;
        }
        offsetData[1] = y * hPiece;
        for (int x = 0; x < sliceNumber; ++x) {
            int wCount = wPiece;
            if (x == sliceNumber - 1) {
                wCount = owUnit - (sliceNumber - 1) * wPiece;
            }
            offsetData[0] = x * wPiece;
            int i         = y * sliceNumber + x;
            mOffsetsBuffer[i].reset(new VulkanBuffer(mBackend->getMemoryPool(), false, sizeof(offsetData), offsetData,
                                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
            mSourceTransformSet[i].reset(mSourceTransform->createSet());
            mDestTransformSet[i].reset(mDestTransform->createSet());
            if (true) {
                auto sourceImage = mMultier->source();
                mSourceTransformSet[i]->writeImage(sourceImage->view(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
                mSourceTransformSet[i]->writeImage((VkImageView)src->deviceId(), mSampler->get(),
                                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
                mSourceTransformSet[i]->writeBuffer(mWinogradConst->buffer(), 2, mWinogradConst->size());
                mSourceTransformSet[i]->writeBuffer(mOffsetsBuffer[i]->buffer(), 3, mOffsetsBuffer[i]->size());
                mSourceTransform->bind(cmdBuffer->get(), mSourceTransformSet[i]->get());
                vkCmdDispatch(cmdBuffer->get(), UP_DIV(wCount, mTransformLocalSize[0]),
                              UP_DIV(hCount, mTransformLocalSize[1]), UP_DIV(icC4, mTransformLocalSize[2]));
            }

            mMultier->compute(cmdBuffer);
            if (true) {
                auto destImage = mMultier->dest();
                mDestTransformSet[i]->writeImage((VkImageView)dst->deviceId(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL,
                                                 0);
                mDestTransformSet[i]->writeImage(destImage->view(), mSampler->get(),
                                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
                mDestTransformSet[i]->writeImage(mBias->view(), mSampler->get(),
                                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
                mDestTransformSet[i]->writeBuffer(mWinogradConst->buffer(), 3, mWinogradConst->size());
                mDestTransformSet[i]->writeBuffer(mOffsetsBuffer[i]->buffer(), 4, mOffsetsBuffer[i]->size());
                mDestTransform->bind(cmdBuffer->get(), mDestTransformSet[i]->get());
                cmdBuffer->barrierImage(destImage->get(), VK_IMAGE_LAYOUT_GENERAL,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                vkCmdDispatch(cmdBuffer->get(), UP_DIV(wCount, mTransformLocalSize[0]),
                              UP_DIV(hCount, mTransformLocalSize[1]), UP_DIV(ocC4, mTransformLocalSize[2]));
            }
        }
    }

    return NO_ERROR;
}

} // namespace MNN
