//
//  VulkanNormlize.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanNormlize.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {
struct GpuParam {
    ivec4 imgSize;
    int channelDiv4;
    float eps;
};

VulkanNormlize::VulkanNormlize(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto normlizeParam = op->main_as_Normalize();
    mEps               = normlizeParam->eps();

    std::vector<VkDescriptorType> VulkanNormlizeTypes{
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };

    std::vector<VkDescriptorType> VulkanScaleTypes{
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};

    mVkBackend = static_cast<VulkanBackend*>(bn);
    mSampler   = mVkBackend->getCommonSampler();
    // normlize
    mVulkanNormlizePipeline =
        mVkBackend->getPipeline("glsl_normalizeChannel_comp",
                                /*glsl_normalizeChannel_comp, glsl_normalizeChannel_comp_len,*/ VulkanNormlizeTypes);
    mParamBuffer.reset(new VulkanBuffer(mVkBackend->getMemoryPool(), false, sizeof(GpuParam), nullptr,
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    MNN_ASSERT(normlizeParam->channelShared() == false);
    // scale
    mVulkanScalePipeline =
        mVkBackend->getPipeline("glsl_scale_comp", /*glsl_scale_comp, glsl_scale_comp_len,*/ VulkanScaleTypes);

    mScale.reset(new VulkanBuffer(mVkBackend->getMemoryPool(), false, sizeof(float) * normlizeParam->scale()->size(),
                                  normlizeParam->scale()->data(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    mBias.reset(new VulkanBuffer(mVkBackend->getMemoryPool(), false, sizeof(float) * normlizeParam->scale()->size(),
                                 nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    auto biasPtr = reinterpret_cast<float*>(mBias->map());
    ::memset(biasPtr, 0, sizeof(float) * normlizeParam->scale()->size());
    mBias->unmap();
}
VulkanNormlize::~VulkanNormlize() {
}
ErrorCode VulkanNormlize::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
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
    auto VulkanNormlizeParam = reinterpret_cast<GpuParam*>(mParamBuffer->map());
    ::memset(VulkanNormlizeParam, 0, sizeof(GpuParam));

    VulkanNormlizeParam->imgSize[0]  = input->width();
    VulkanNormlizeParam->imgSize[1]  = input->height();
    VulkanNormlizeParam->imgSize[2]  = channelDiv4;
    VulkanNormlizeParam->imgSize[3]  = 0;
    VulkanNormlizeParam->channelDiv4 = channelDiv4;
    VulkanNormlizeParam->eps         = mEps;

    mParamBuffer->flush(true, 0, sizeof(GpuParam));
    mParamBuffer->unmap();

    // normlize
    mNormlizeDescriptorSet.reset(mVulkanNormlizePipeline->createSet());
    mNormlizeDescriptorSet->writeImage(reinterpret_cast<VkImageView>(mTempTensor.deviceId()), mSampler->get(),
                                       VK_IMAGE_LAYOUT_GENERAL, 0);
    mNormlizeDescriptorSet->writeImage(reinterpret_cast<VkImageView>(input->deviceId()), mSampler->get(),
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mNormlizeDescriptorSet->writeBuffer(mParamBuffer->buffer(), 2, mParamBuffer->size());

    mVulkanNormlizePipeline->bind(cmdBuffer->get(), mNormlizeDescriptorSet->get());

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

class VulkanNormlizeCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanNormlize(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Normalize, new VulkanNormlizeCreator);
    return true;
}();

} // namespace MNN
