//
//  VulkanScale.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanScale.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

struct gpuScaleParam {
    ivec4 imgSize;
};

VulkanScale::VulkanScale(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    const auto scale   = op->main_as_Scale();
    const int channels = scale->scaleData()->size();

    std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};

    auto extra = static_cast<VulkanBackend*>(bn);

    mScalePipeline = extra->getPipeline("glsl_scale_comp", /*glsl_scale_comp, glsl_scale_comp_len,*/ types);
    mScaleParam    = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false, sizeof(gpuScaleParam), nullptr,
                                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    mScaleBuffer   = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false, sizeof(float) * channels,
                                                  scale->scaleData()->data(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    mBiasBuffer    = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false, sizeof(float) * channels,
                                                 scale->biasData()->data(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    mSampler       = extra->getCommonSampler();
}

VulkanScale::~VulkanScale() {
}

ErrorCode VulkanScale::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];

    MNN_ASSERT(MNN_DATA_FORMAT_NC4HW4 == TensorUtils::getDescribe(input)->dimensionFormat);

    auto scaleP = reinterpret_cast<gpuScaleParam*>(mScaleParam->map());
    ::memset(scaleP, 0, sizeof(gpuScaleParam));

    const int channelDiv4 = UP_DIV(input->channel(), 4);

    scaleP->imgSize[0] = input->width();
    scaleP->imgSize[1] = input->height();
    scaleP->imgSize[2] = channelDiv4;
    scaleP->imgSize[3] = input->batch();
    mScaleParam->flush(true, 0, sizeof(gpuScaleParam));
    mScaleParam->unmap();

    mDescriptorSet.reset(mScalePipeline->createSet());
    mDescriptorSet->writeImage(reinterpret_cast<VulkanTensor*>(output->deviceId())->image()->view(), mSampler->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 0);
    mDescriptorSet->writeImage(reinterpret_cast<VulkanTensor*>(input->deviceId())->image()->view(), mSampler->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mDescriptorSet->writeBuffer(mScaleBuffer->buffer(), 2, mScaleBuffer->size());
    mDescriptorSet->writeBuffer(mBiasBuffer->buffer(), 3, mBiasBuffer->size());
    mDescriptorSet->writeBuffer(mScaleParam->buffer(), 4, mScaleParam->size());
    mScalePipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

    reinterpret_cast<VulkanTensor*>(output->deviceId())->image()->barrierWrite(cmdBuffer->get());
    reinterpret_cast<VulkanTensor*>(input->deviceId())->image()->barrierRead(cmdBuffer->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(input->width(), 16), UP_DIV(input->height(), 16),
                  channelDiv4 * input->batch());

    return NO_ERROR;
}

class VulkanScaleCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanScale(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Scale, new VulkanScaleCreator);
    return true;
}();

} // namespace MNN
