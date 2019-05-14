//
//  VulkanGroupConvolution.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanGroupConvolution.hpp"
#include "ConvolutionIntFactory.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"
namespace MNN {
VulkanGroupConvolution::VulkanGroupConvolution(const Op *op, Backend *backend)
    : Execution(backend), mTempSrc(4), mTempDst(4) {
    mConvParameter = op->main_as_Convolution2D();
    mBackend       = static_cast<VulkanBackend *>(backend);
}

VulkanGroupConvolution::~VulkanGroupConvolution() {
}
ErrorCode VulkanGroupConvolution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    for (auto &iter : mSubConvolutions) {
        mBackend->pushCommand(std::get<0>(iter)->get());
        std::get<1>(iter)->onExecute(mTempInputs, mTempOutputs);
        mBackend->pushCommand(std::get<2>(iter)->get());
    }
    return NO_ERROR;
}

ErrorCode VulkanGroupConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input      = inputs[0];
    auto output     = outputs[0];
    const int group = mConvParameter->common()->group();
    mTempInputs     = std::vector<Tensor *>{&mTempSrc};
    mTempOutputs    = std::vector<Tensor *>{&mTempDst};
    if (mSubConvolutions.empty()) {
        mSubConvolutions.resize(group);
        const auto convReal    = mConvParameter;
        const auto common      = convReal->common();
        const auto outputCount = common->outputCount();
        const int fh           = common->kernelY();
        const int fw           = common->kernelX();
        int groupCI            = 0;
        const float *source    = nullptr;
        std::shared_ptr<ConvolutionIntFactory::Int8Common> quanCommon;
        // check whether idst quantized op
        if (nullptr != convReal->quanParameter()) {
            quanCommon = ConvolutionIntFactory::load(convReal->quanParameter(), true);
            groupCI    = quanCommon->weightFloat.size() / (outputCount * fh * fw);
            source     = quanCommon->weightFloat.get();
        } else {
            groupCI = convReal->weight()->size() / (outputCount * fh * fw);
            source  = convReal->weight()->data();
        }

        const int groupCO         = outputCount / group;
        const int groupWeightSize = groupCI * fw * fh * groupCO;

        for (int i = 0; i < group; ++i) {
            const float *curWeightPtr = source + i * groupWeightSize;
            const float *curBiasPtr   = convReal->bias()->data() + i * groupCO;
            std::shared_ptr<Execution> subConvolution(VulkanConvolutionImpl::create(
                mBackend, mConvParameter->common(), input, output, curWeightPtr, curBiasPtr, groupCI, groupCO));
            std::get<1>(mSubConvolutions[i]) = subConvolution;
        }
    }

    // copy input-output's shape and acquire memory
    TensorUtils::copyShape(input, &mTempSrc, true);
    mTempSrc.setLength(1, input->channel() / group);
    TensorUtils::copyShape(output, &mTempDst, true);
    mTempDst.setLength(1, output->channel() / group);
    backend()->onAcquireBuffer(&mTempSrc, Backend::DYNAMIC);
    backend()->onAcquireBuffer(&mTempDst, Backend::DYNAMIC);

    auto inputImage  = mBackend->findTensor(input->deviceId())->image()->get();
    auto outputImage = mBackend->findTensor(output->deviceId())->image()->get();

    auto tempSrcImage = mBackend->findTensor(mTempSrc.deviceId())->image()->get();
    auto tempDstImage = mBackend->findTensor(mTempDst.deviceId())->image()->get();

    const int tempChannelDiv4Src = mTempSrc.channel() / 4;
    const int tempChannelDiv4Dst = mTempDst.channel() / 4;
    VkImageCopy copyRegionInput;
    ::memset(&copyRegionInput, 0, sizeof(copyRegionInput));
    copyRegionInput.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegionInput.srcSubresource.mipLevel       = 0;
    copyRegionInput.srcSubresource.baseArrayLayer = 0;
    copyRegionInput.srcSubresource.layerCount     = 1;
    copyRegionInput.srcOffset.x                   = 0;
    copyRegionInput.srcOffset.y                   = 0;
    copyRegionInput.srcOffset.z                   = 0;

    copyRegionInput.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegionInput.dstSubresource.mipLevel       = 0;
    copyRegionInput.dstSubresource.baseArrayLayer = 0;
    copyRegionInput.dstSubresource.layerCount     = 1;
    copyRegionInput.dstOffset.x                   = 0;
    copyRegionInput.dstOffset.y                   = 0;
    copyRegionInput.dstOffset.z                   = 0;
    copyRegionInput.extent.width                  = mTempSrc.width();
    copyRegionInput.extent.height                 = mTempSrc.height();
    copyRegionInput.extent.depth                  = tempChannelDiv4Src;

    VkImageCopy copyRegionOutput;
    ::memset(&copyRegionOutput, 0, sizeof(copyRegionOutput));
    copyRegionOutput.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegionOutput.srcSubresource.mipLevel       = 0;
    copyRegionOutput.srcSubresource.baseArrayLayer = 0;
    copyRegionOutput.srcSubresource.layerCount     = 1;
    copyRegionOutput.srcOffset.x                   = 0;
    copyRegionOutput.srcOffset.y                   = 0;
    copyRegionOutput.srcOffset.z                   = 0;

    copyRegionOutput.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegionOutput.dstSubresource.mipLevel       = 0;
    copyRegionOutput.dstSubresource.baseArrayLayer = 0;
    copyRegionOutput.dstSubresource.layerCount     = 1;
    copyRegionOutput.dstOffset.x                   = 0;
    copyRegionOutput.dstOffset.y                   = 0;
    copyRegionOutput.dstOffset.z                   = 0;
    copyRegionOutput.extent.width                  = mTempDst.width();
    copyRegionOutput.extent.height                 = mTempDst.height();
    copyRegionOutput.extent.depth                  = tempChannelDiv4Dst;

    for (int i = 0; i < group; ++i) {
        {
            copyRegionInput.srcOffset.z = i * tempChannelDiv4Src;
            std::get<0>(mSubConvolutions[i])
                .reset(const_cast<VulkanCommandPool::Buffer *>(mBackend->getPool().allocBuffer()));
            auto cmdBuffer = std::get<0>(mSubConvolutions[i]).get();
            cmdBuffer->begin(0);
            vkCmdCopyImage(cmdBuffer->get(), inputImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tempSrcImage,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegionInput);
            cmdBuffer->end();
        }
        std::get<1>(mSubConvolutions[i])->onResize(mTempInputs, mTempOutputs);
        {
            copyRegionOutput.dstOffset.z = i * tempChannelDiv4Dst;
            std::get<2>(mSubConvolutions[i])
                .reset(const_cast<VulkanCommandPool::Buffer *>(mBackend->getPool().allocBuffer()));
            auto cmdBuffer = std::get<2>(mSubConvolutions[i]).get();
            cmdBuffer->begin(0);
            vkCmdCopyImage(cmdBuffer->get(), tempDstImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, outputImage,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegionOutput);
            cmdBuffer->end();
        }
    }
    backend()->onReleaseBuffer(&mTempSrc, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mTempDst, Backend::DYNAMIC);
    return NO_ERROR;
}
} // namespace MNN
