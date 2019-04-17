//
//  VulkanRelu.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanRelu.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"
namespace MNN {

struct GpuReluParam {
    ivec4 imgSize;
    float slope;
};

//--------------------------relu--------------------------//
VulkanRelu::VulkanRelu(Backend *bn, float slope) : VulkanBasicExecution(bn), mSlope(slope) {
    std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    auto vulkanBn = static_cast<VulkanBackend *>(bn);
    mReluPipeline = vulkanBn->getPipeline("glsl_relu_comp", /*glsl_relu_comp, glsl_relu_comp_len,*/ types);
    mGpuReluParam.reset(new VulkanBuffer(vulkanBn->getMemoryPool(), false, sizeof(GpuReluParam), nullptr,
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
}

VulkanRelu::~VulkanRelu() {
}

ErrorCode VulkanRelu::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                               const VulkanCommandPool::Buffer *cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];

    auto reluParam = reinterpret_cast<GpuReluParam *>(mGpuReluParam->map());
    ::memset(reluParam, 0, sizeof(GpuReluParam));

    const int channelDiv4 = UP_DIV(input->channel(), 4);
    reluParam->imgSize[0] = input->width();
    reluParam->imgSize[1] = input->height();
    reluParam->imgSize[2] = channelDiv4 * input->batch();
    reluParam->imgSize[3] = 0;
    reluParam->slope      = mSlope;
    mGpuReluParam->flush(true, 0, sizeof(GpuReluParam));
    mGpuReluParam->unmap();

    auto vkBn = (VulkanBackend *)backend();
    mDescriptorSet.reset(mReluPipeline->createSet());
    mDescriptorSet->writeImage((VkImageView)output->deviceId(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 0);
    mDescriptorSet->writeImage((VkImageView)input->deviceId(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mDescriptorSet->writeBuffer(mGpuReluParam->buffer(), 2, mGpuReluParam->size());

    mReluPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(input->width(), 16), UP_DIV(input->height(), 16),
                  channelDiv4 * input->batch());

    return NO_ERROR;
}
//--------------------------relu6--------------------------//
VulkanRelu6::VulkanRelu6(Backend *bn) : VulkanBasicExecution(bn) {
    std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    auto vulkanBn  = static_cast<VulkanBackend *>(bn);
    mRelu6Pipeline = vulkanBn->getPipeline("glsl_relu6_comp", /*glsl_relu6_comp, glsl_relu6_comp_len,*/ types);
    mGpuRelu6Param.reset(new VulkanBuffer(vulkanBn->getMemoryPool(), false, sizeof(GpuReluParam), nullptr,
                                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
}

VulkanRelu6::~VulkanRelu6() {
}

ErrorCode VulkanRelu6::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const VulkanCommandPool::Buffer *cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];

    auto reluParam = reinterpret_cast<GpuReluParam *>(mGpuRelu6Param->map());
    ::memset(reluParam, 0, sizeof(GpuReluParam));

    const int channelDiv4 = UP_DIV(input->channel(), 4);
    reluParam->imgSize[0] = input->width();
    reluParam->imgSize[1] = input->height();
    reluParam->imgSize[2] = channelDiv4 * input->batch();
    reluParam->imgSize[3] = 0;
    reluParam->slope      = 0;
    mGpuRelu6Param->flush(true, 0, sizeof(GpuReluParam));
    mGpuRelu6Param->unmap();

    auto vkBn = (VulkanBackend *)backend();
    mDescriptorSet.reset(mRelu6Pipeline->createSet());
    mDescriptorSet->writeImage((VkImageView)output->deviceId(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 0);
    mDescriptorSet->writeImage((VkImageView)input->deviceId(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mDescriptorSet->writeBuffer(mGpuRelu6Param->buffer(), 2, mGpuRelu6Param->size());

    mRelu6Pipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(input->width(), 16), UP_DIV(input->height(), 16),
                  channelDiv4 * input->batch());

    return NO_ERROR;
}
//--------------------------Prelu--------------------------//
VulkanPrelu::VulkanPrelu(Backend *bn, const Op *op) : VulkanBasicExecution(bn) {
    std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    auto vulkanBn    = static_cast<VulkanBackend *>(bn);
    mPreluPipeline   = vulkanBn->getPipeline("glsl_preluWithChannel_comp",
                                           /*glsl_preluWithChannel_comp, glsl_preluWithChannel_comp_len,*/ types);
    const auto prelu = op->main_as_PRelu();
    mGpuPreluParam.reset(new VulkanBuffer(vulkanBn->getMemoryPool(), false, sizeof(GpuReluParam), nullptr,
                                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    int count = ALIGN_UP4(prelu->slope()->size());

    mSlope.reset(new VulkanImage(vulkanBn->getMemoryPool(), false, std::vector<int>{count / 4, 1}));
    {
        std::shared_ptr<VulkanBuffer> slopeBuffer(new VulkanBuffer(
            vulkanBn->getMemoryPool(), false, sizeof(float) * count, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        auto slope = slopeBuffer->map();
        ::memset(slope, 0, count * sizeof(float));
        ::memcpy(slope, prelu->slope()->data(), prelu->slope()->size() * sizeof(float));
        slopeBuffer->unmap();
        vulkanBn->copyBufferToImage(slopeBuffer.get(), mSlope.get());
    }
}

VulkanPrelu::~VulkanPrelu() {
}

ErrorCode VulkanPrelu::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const VulkanCommandPool::Buffer *cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];

    auto preluParam = reinterpret_cast<GpuReluParam *>(mGpuPreluParam->map());
    ::memset(preluParam, 0, sizeof(GpuReluParam));
    auto vkBn = static_cast<VulkanBackend *>(backend());

    const int channelDiv4  = UP_DIV(input->channel(), 4);
    preluParam->imgSize[0] = input->width();
    preluParam->imgSize[1] = input->height();
    preluParam->imgSize[2] = channelDiv4;
    preluParam->imgSize[3] = 0;
    mGpuPreluParam->flush(true, 0, sizeof(GpuReluParam));
    mGpuPreluParam->unmap();

    mDescriptorSet.reset(mPreluPipeline->createSet());
    mDescriptorSet->writeImage((VkImageView)output->deviceId(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 0);
    mDescriptorSet->writeImage((VkImageView)input->deviceId(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mDescriptorSet->writeImage((VkImageView)mSlope->view(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
    mDescriptorSet->writeBuffer(mGpuPreluParam->buffer(), 3, mGpuPreluParam->size());

    mPreluPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(input->width(), 16), UP_DIV(input->height(), 16), channelDiv4);
    return NO_ERROR;
}

class VulkanReluCreator : public VulkanBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *bn) const override {
        auto type  = op->type();
        auto input = inputs[0];
        if (TensorUtils::getDescribe(input)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
            return nullptr;
        }
        if (OpType_ReLU6 == type) {
            return new VulkanRelu6(bn);
        }
        if (OpType_ReLU == type) {
            return new VulkanRelu(bn, op->main_as_Relu()->slope());
        } else if (1 == op->main_as_PRelu()->slopeCount()) {
            return new VulkanRelu(bn, op->main_as_PRelu()->slope()->data()[0]);
        } else {
            return new VulkanPrelu(bn, op);
        }
    }
};

static bool gr = []() {
    VulkanBackend::addCreator(OpType_ReLU, new VulkanReluCreator);
    VulkanBackend::addCreator(OpType_PReLU, new VulkanReluCreator);
    VulkanBackend::addCreator(OpType_ReLU6, new VulkanReluCreator);
    return true;
}();

} // namespace MNN
