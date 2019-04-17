//
//  VulkanConvolution.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanConvolution.hpp"
#include "Macro.h"
#include "VulkanConvolutionWrap.hpp"
#include "VulkanGroupConvolution.hpp"
//#define MNN_USE_1x1
namespace MNN {
std::string VulkanConvolutionCommon::getPostTreatMacro(const Convolution2DCommon* common) {
    if (common->relu()) {
        return "RELU_";
    } else if (common->relu6()) {
        return "RELU6_";
    }
    return "";
}

static std::shared_ptr<VulkanBuffer> _createBufferForConvDepthwise(VulkanBackend* extra,
                                                                   const Convolution2DCommon* mCommon,
                                                                   const float* weightSource, size_t weightSize) {
    auto outputCount     = mCommon->outputCount();
    auto totalWeightSize = ALIGN_UP4(mCommon->outputCount()) * (mCommon->kernelY() * mCommon->kernelX());
    auto kernelBuffer = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false, sizeof(float) * totalWeightSize);
    auto layer        = mCommon;

    auto weight     = (float*)kernelBuffer->map();
    int kw          = layer->kernelX();
    int kh          = layer->kernelY();
    int planeStride = kw * kh * 4;

    int cur = 0;
    for (int c = 0; c < outputCount; ++c) {
        int plane  = c / 4;
        int offset = c % 4;
        for (int y = 0; y < kh; ++y) {
            for (int x = 0; x < kw; ++x) {
                float* dst = weight + offset + (x + y * kw) * 4 + planeStride * plane;
                *dst       = weightSource[cur++];
            }
        }
    }
    kernelBuffer->unmap();
    return kernelBuffer;
}

void VulkanConvolutionCommon::writeParameter(ConvolutionParameter* convCons, const Convolution2DCommon* common,
                                             const Tensor* input, const Tensor* output) {
    int icDiv4 = UP_DIV(input->channel(), 4);
    int ocDiv4 = UP_DIV(output->channel(), 4);
    int padX   = common->padX();
    int padY   = common->padY();
    if (common->padMode() == PadMode_SAME) {
        int kernelWidthSize = (common->kernelX() - 1) * common->dilateX() + 1;
        int kernelHeightSize = (common->kernelY() - 1) * common->dilateY() + 1;
        int pad_needed_width  = (output->width() - 1) * common->strideX() + kernelWidthSize - input->width();
        int pad_needed_height = (output->height() - 1) * common->strideY() + kernelHeightSize - input->height();
        padX = pad_needed_width / 2;
        padY = pad_needed_height / 2;
    }
    {
        convCons->batch         = input->batch();
        convCons->dilate[0]     = common->dilateX();
        convCons->dilate[1]     = common->dilateY();
        convCons->stride[0]     = common->strideX();
        convCons->stride[1]     = common->strideY();
        convCons->pad[0]        = padX;
        convCons->pad[1]        = padY;
        convCons->kernelSize[0] = common->kernelX();
        convCons->kernelSize[1] = common->kernelY();

        convCons->inputSize[0] = input->width();
        convCons->inputSize[1] = input->height();
        convCons->inputSize[2] = icDiv4;
        convCons->inputSize[3] = input->batch();

        convCons->outputSize[0] = output->width();
        convCons->outputSize[1] = output->height();
        convCons->outputSize[2] = ocDiv4;
        convCons->outputSize[3] = output->batch();
        convCons->group         = common->group();
    }
}

VulkanConvolutionCommon::VulkanConvolutionCommon(const Op* convOp, Backend* bn) : VulkanBasicExecution(bn) {
    auto extra    = static_cast<VulkanBackend*>(bn);
    mCommon       = convOp->main_as_Convolution2D()->common();
    auto convReal = convOp->main_as_Convolution2D();

    // Create Buffer
    auto biasBuffer = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false,
                                                     sizeof(float) * ALIGN_UP4(mCommon->outputCount()));

    auto bias = biasBuffer->map();
    ::memset(bias, 0, ALIGN_UP4(mCommon->outputCount()) * sizeof(float));
    ::memcpy(bias, convReal->bias()->data(), convReal->bias()->size() * sizeof(float));
    biasBuffer->unmap();

    mBias = std::make_shared<VulkanImage>(extra->getMemoryPool(), false, UP_DIV(mCommon->outputCount(), 4), 1);
    extra->copyBufferToImage(biasBuffer.get(), mBias.get());
    mConvCons = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false, sizeof(ConvolutionParameter), nullptr,
                                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
}

VulkanConvolutionCommon::~VulkanConvolutionCommon() {
}

ErrorCode VulkanConvolutionCommon::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                            const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];
    {
        auto convCons = (ConvolutionParameter*)mConvCons->map();
        writeParameter(convCons, mCommon, input, output);
        mConvCons->unmap();
    }

    auto code = this->onEncodeConvolution(mCommon, inputs, outputs, cmdBuffer, mConvCons.get(), mBias.get());
    if (NO_ERROR != code) {
        return code;
    }
    return NO_ERROR;
}

VulkanConvolutionDepthwise::VulkanConvolutionDepthwise(const Op* convOp, Backend* bn)
    : VulkanConvolutionCommon(convOp, bn) {
    auto extra      = static_cast<VulkanBackend*>(bn);
    auto mCommon    = convOp->main_as_Convolution2D()->common();
    auto convReal   = convOp->main_as_Convolution2D();
    auto weightSize = convReal->weight()->size();
    mSampler        = extra->getCommonSampler();
    // Create Pipeline
    std::vector<VkDescriptorType> convTypes{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    MNN_ASSERT(OpType_ConvolutionDepthwise == convOp->type());
    auto macro = getPostTreatMacro(mCommon);
    if (extra->gpuType() == VulkanBackend::ADRENO) {
        mConvPipeline = extra->getPipeline("glsl_convolutionDepthwise_" + macro + "comp", convTypes);
        mLocalX       = 16;
        mLocalY       = 16;
    } else {
        mConvPipeline = extra->getPipeline("glsl_convolutionDepthwiseMali_" + macro + "comp", convTypes);
        mLocalX       = 8;
        mLocalY       = 8;
    }

    auto kernelBuffer = _createBufferForConvDepthwise(extra, mCommon, convReal->weight()->data(), weightSize);
    mKernel = std::make_shared<VulkanImage>(extra->getMemoryPool(), false, mCommon->kernelX() * mCommon->kernelY(),
                                            UP_DIV(mCommon->outputCount(), 4));
    extra->copyBufferToImage(kernelBuffer.get(), mKernel.get());
}

VulkanConvolutionDepthwise::~VulkanConvolutionDepthwise() {
}

ErrorCode VulkanConvolutionDepthwise::onEncodeConvolution(const Convolution2DCommon* common,
                                                          const std::vector<Tensor*>& inputs,
                                                          const std::vector<Tensor*>& outputs,
                                                          const VulkanCommandPool::Buffer* cmdBuffer,
                                                          const VulkanBuffer* convCons, const VulkanImage* biasBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];
    /*Set Const Parameters*/
    int ocDiv4 = UP_DIV(output->channel(), 4);
    int ow     = output->width();
    int oh     = output->height();

    /*Write Command Buffer*/
    if (true) {
        mConvSet.reset(mConvPipeline->createSet());
        mConvSet->writeImage((VkImageView)output->deviceId(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
        mConvSet->writeImage((VkImageView)input->deviceId(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             1);
        mConvSet->writeImage(mKernel->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
        mConvSet->writeImage(biasBuffer->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 3);
        mConvSet->writeBuffer(convCons->buffer(), 4, convCons->size());
        mConvPipeline->bind(cmdBuffer->get(), mConvSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(ow, mLocalX), UP_DIV(oh, mLocalY), ocDiv4 * input->batch());
    }
    return NO_ERROR;
}

class VulkanConvolutionCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op,
                                Backend* backend) const override {
        if (op->type() == OpType_Convolution) {
            auto convCommonParam = op->main_as_Convolution2D()->common();
            const int group      = convCommonParam->group();
            if (1 == group) {
                return new VulkanConvolutionWrap(op, backend);
            } else {
                return new VulkanGroupConvolution(op, backend);
            }
        }
        return new VulkanConvolutionDepthwise(op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Convolution, new VulkanConvolutionCreator);
    VulkanBackend::addCreator(OpType_ConvolutionDepthwise, new VulkanConvolutionCreator);
    return true;
}();

} // namespace MNN
