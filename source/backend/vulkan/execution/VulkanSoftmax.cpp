//
//  VulkanSoftmax.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanSoftmax.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {

struct ConstBuffer {
    int w;
    int h;
    int c;
};

VulkanSoftmax::VulkanSoftmax(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    const auto softmaxParam = op->main_as_Axis();
    mAxis                   = softmaxParam->axis();

    mVkBackend = static_cast<VulkanBackend*>(bn);

    mConstBuffer = std::make_shared<VulkanBuffer>(mVkBackend->getMemoryPool(), false, sizeof(ConstBuffer), nullptr,
                                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
}

VulkanSoftmax::~VulkanSoftmax() {
}

ErrorCode VulkanSoftmax::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];

    auto inputFormat = TensorUtils::getDescribe(input)->dimensionFormat;
    if (mAxis < 0) {
        mAxis = input->dimensions() + mAxis;
    }
    if (MNN_DATA_FORMAT_NHWC == inputFormat) {
        // for NHWC input
        std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        if (1 == mAxis) {
            mSoftmaxPipeline =
                mVkBackend->getPipeline("glsl_softmaxHeight_NHWC_comp",
                                        /*glsl_softmaxHeight_NHWC_comp, glsl_softmaxHeight_NHWC_comp_len,*/ types);
        } else {
            MNN_ASSERT(false);
        }

        // gpu param
        const int height  = std::max(1, input->height());
        const int width   = std::max(1, input->width());
        const int channel = std::max(1, input->channel());
        {
            auto softmax = reinterpret_cast<ConstBuffer*>(mConstBuffer->map());
            ::memset(softmax, 0, sizeof(ConstBuffer));
            softmax->w = width;
            softmax->h = height;
            softmax->c = channel;
            mConstBuffer->flush(true, 0, sizeof(ConstBuffer));
            mConstBuffer->unmap();
        }
        mDescriptorSet.reset(mSoftmaxPipeline->createSet());
        mDescriptorSet->writeBuffer(reinterpret_cast<VkBuffer>(output->deviceId()), 0, output->size());
        mDescriptorSet->writeBuffer(reinterpret_cast<VkBuffer>(input->deviceId()), 1, input->size());
        mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 2, mConstBuffer->size());
        mSoftmaxPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
        cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(input->deviceId()), 0, input->size());
        // dispatch
        if (1 == mAxis) {
            vkCmdDispatch(cmdBuffer->get(), channel, width, 1);
        }
    } else {
        // NC4HW4 input
        std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};

        if (1 == mAxis) {
            mSoftmaxPipeline = mVkBackend->getPipeline(
                "glsl_softmaxChannel_comp", /*glsl_softmaxChannel_comp, glsl_softmaxChannel_comp_len,*/ types);
        } else if (2 == mAxis) {
            mSoftmaxPipeline = mVkBackend->getPipeline("glsl_softmaxHeight_comp",
                                                       /*glsl_softmaxHeight_comp, glsl_softmaxHeight_comp_len,*/ types);
        } else if (3 == mAxis) {
            mSoftmaxPipeline = mVkBackend->getPipeline("glsl_softmaxWidth_comp",
                                                       /*glsl_softmaxWidth_comp, glsl_softmaxWidth_comp_len,*/ types);
        } else {
            MNN_ASSERT(false);
        }

        const int channelsDiv4 = UP_DIV(input->channel(), 4);
        const int width        = std::max(1, input->width());
        const int height       = std::max(1, input->height());

        {
            auto softmax = reinterpret_cast<ConstBuffer*>(mConstBuffer->map());
            ::memset(softmax, 0, sizeof(ConstBuffer));
            softmax->w = width;
            softmax->h = height;
            softmax->c = input->channel();
            mConstBuffer->flush(true, 0, sizeof(ConstBuffer));
            mConstBuffer->unmap();
        }

        mDescriptorSet.reset(mSoftmaxPipeline->createSet());
        mDescriptorSet->writeImage(reinterpret_cast<VkImageView>(output->deviceId()),
                                   mVkBackend->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
        mDescriptorSet->writeImage(reinterpret_cast<VkImageView>(input->deviceId()),
                                   mVkBackend->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 2, mConstBuffer->size());
        mSoftmaxPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

        if (1 == mAxis) {
            vkCmdDispatch(cmdBuffer->get(), UP_DIV(width, 8), UP_DIV(height, 8), input->batch());
        } else if (2 == mAxis) {
            vkCmdDispatch(cmdBuffer->get(), UP_DIV(width, 8), 1, channelsDiv4 * input->batch());
        } else {
            vkCmdDispatch(cmdBuffer->get(), 1, UP_DIV(width, 8), channelsDiv4 * input->batch());
        }
    }

    return NO_ERROR;
}

class VulkanSoftmaxCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new VulkanSoftmax(op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Softmax, new VulkanSoftmaxCreator);
    return true;
}();

} // namespace MNN
