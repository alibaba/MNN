//
//  VulkanNormalize.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanNormalize.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {
struct GpuParam {
    ivec4 imgSize;
    int channelDiv4;
    float eps;
};

VulkanNormalize::VulkanNormalize(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto normalizeParam = op->main_as_Normalize();
    mEps                = normalizeParam->eps();

    std::vector<VkDescriptorType> VulkanNormalizeTypes{
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };

    std::vector<VkDescriptorType> VulkanScaleTypes{
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};

    mVkBackend = static_cast<VulkanBackend*>(bn);
    mSampler   = mVkBackend->getCommonSampler();
    // normalize
    mVulkanNormalizePipeline =
        mVkBackend->getPipeline("glsl_normalizeChannel_comp",
                                /*glsl_normalizeChannel_comp, glsl_normalizeChannel_comp_len,*/ VulkanNormalizeTypes);
    mParamBuffer.reset(new VulkanBuffer(mVkBackend->getMemoryPool(), false, sizeof(GpuParam), nullptr,
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    MNN_ASSERT(normalizeParam->channelShared() == false);
    // scale
    mVulkanScalePipeline =
        mVkBackend->getPipeline("glsl_scale_comp", /*glsl_scale_comp, glsl_scale_comp_len,*/ VulkanScaleTypes);

    mScale.reset(new VulkanBuffer(mVkBackend->getMemoryPool(), false, sizeof(float) * normalizeParam->scale()->size(),
                                  normalizeParam->scale()->data(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    mBias.reset(new VulkanBuffer(mVkBackend->getMemoryPool(), false, sizeof(float) * normalizeParam->scale()->size(),
                                 nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    auto biasPtr = reinterpret_cast<float*>(mBias->map());
    ::memset(biasPtr, 0, sizeof(float) * normalizeParam->scale()->size());
    mBias->unmap();
}
VulkanNormalize::~VulkanNormalize() {
}
ErrorCode VulkanNormalize::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input            = inputs[0];
    auto output           = outputs[0];
    const int channelDiv4 = UP_DIV(input->channel(), 4);

    TensorUtils::copyShape(input, &mTempTensor, true);
    MNN_ASSERT(TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4);
    backend()->onAcquireBuffer(&mTempTensor, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mTempTensor, Backend::DYNAMIC);

    auto tempTensorImage = mVkBackend->findTensor(mTempTensor.deviceId())->image();
    MNN_ASSERT(nullptr != tempTensorImage);
    auto VulkanNormalizeParam = reinterpret_cast<GpuParam*>(mParamBuffer->map());
    ::memset(VulkanNormalizeParam, 0, sizeof(GpuParam));

    VulkanNormalizeParam->imgSize[0]  = input->width();
    VulkanNormalizeParam->imgSize[1]  = input->height();
    VulkanNormalizeParam->imgSize[2]  = channelDiv4;
    VulkanNormalizeParam->imgSize[3]  = 0;
    VulkanNormalizeParam->channelDiv4 = channelDiv4;
    VulkanNormalizeParam->eps         = mEps;

    mParamBuffer->flush(true, 0, sizeof(GpuParam));
    mParamBuffer->unmap();

    // normalize
    mNormalizeDescriptorSet.reset(mVulkanNormalizePipeline->createSet());
    mNormalizeDescriptorSet->writeImage(reinterpret_cast<VkImageView>(mTempTensor.deviceId()), mSampler->get(),
                                        VK_IMAGE_LAYOUT_GENERAL, 0);
    mNormalizeDescriptorSet->writeImage(reinterpret_cast<VkImageView>(input->deviceId()), mSampler->get(),
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mNormalizeDescriptorSet->writeBuffer(mParamBuffer->buffer(), 2, mParamBuffer->size());

    mVulkanNormalizePipeline->bind(cmdBuffer->get(), mNormalizeDescriptorSet->get());

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(input->width(), 8), UP_DIV(input->height(), 8), input->batch());

    // scale
    mScaleDescriptorSet.reset(mVulkanScalePipeline->createSet());
    mScaleDescriptorSet->writeImage(reinterpret_cast<VkImageView>(output->deviceId()), mSampler->get(),
                                    VK_IMAGE_LAYOUT_GENERAL, 0);
    mScaleDescriptorSet->writeImage(reinterpret_cast<VkImageView>(mTempTensor.deviceId()), mSampler->get(),
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mScaleDescriptorSet->writeBuffer(mScale->buffer(), 2, mScale->size());
    mScaleDescriptorSet->writeBuffer(mBias->buffer(), 3, mBias->size());
    mScaleDescriptorSet->writeBuffer(mParamBuffer->buffer(), 4, mParamBuffer->size());
    mVulkanScalePipeline->bind(cmdBuffer->get(), mScaleDescriptorSet->get());

    cmdBuffer->barrierImage(tempTensorImage->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(input->width(), 16), UP_DIV(input->height(), 16),
                  channelDiv4 * input->batch());

    return NO_ERROR;
}

class VulkanNormalizeCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanNormalize(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Normalize, new VulkanNormalizeCreator);
    return true;
}();

} // namespace MNN
