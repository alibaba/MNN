//
//  VulkanResize.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanResize.hpp"
#include "Macro.h"

namespace MNN {
struct GpuParam {
    ivec4 inImgSize;
    ivec4 outImgSize;
    vec2 scale;
};

VulkanResize::VulkanResize(Backend* bn, float xScale, float yScale, int resizeType)
    : VulkanBasicExecution(bn), mXScale(xScale), mYScale(yScale) {
    std::vector<VkDescriptorType> VulkanResizeTypes{
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    auto extra            = static_cast<VulkanBackend*>(bn);
    if (1 == resizeType) {
        mVulkanResizePipeline = extra->getPipeline(
                                                   "glsl_resizeNearest_comp", VulkanResizeTypes);
    } else {
        mVulkanResizePipeline = extra->getPipeline(
                                                   "glsl_resizeBilinear_comp", VulkanResizeTypes);
    }
    mParamBuffer.reset(
        new VulkanBuffer(extra->getMemoryPool(), false, sizeof(GpuParam), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
}
VulkanResize::~VulkanResize() {
}

ErrorCode VulkanResize::encodeImpl(Tensor* input, Tensor* output, float xScale, float yScale,
                                   const VulkanCommandPool::Buffer* cmdBuffer) {
    const int channelDiv4 = UP_DIV(input->channel(), 4);
    auto extra            = static_cast<VulkanBackend*>(backend());

    auto VulkanResizeParam = reinterpret_cast<GpuParam*>(mParamBuffer->map());
    ::memset(VulkanResizeParam, 0, sizeof(GpuParam));
    VulkanResizeParam->inImgSize[0]  = input->width();
    VulkanResizeParam->inImgSize[1]  = input->height();
    VulkanResizeParam->inImgSize[2]  = channelDiv4;
    VulkanResizeParam->inImgSize[3]  = input->batch();
    VulkanResizeParam->outImgSize[0] = output->width();
    VulkanResizeParam->outImgSize[1] = output->height();
    VulkanResizeParam->outImgSize[2] = channelDiv4;
    VulkanResizeParam->outImgSize[3] = output->batch();
    VulkanResizeParam->scale[0]      = xScale;
    VulkanResizeParam->scale[1]      = yScale;
    mParamBuffer->flush(true, 0, sizeof(GpuParam));
    mParamBuffer->unmap();

    mDescriptorSet.reset(mVulkanResizePipeline->createSet());
    mDescriptorSet->writeImage((VkImageView)input->deviceId(), extra->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0);
    mDescriptorSet->writeImage((VkImageView)output->deviceId(), extra->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 1);
    mDescriptorSet->writeBuffer(mParamBuffer->buffer(), 2, mParamBuffer->size());
    mVulkanResizePipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(output->width(), 16), UP_DIV(output->height(), 16),
                  channelDiv4 * input->batch());

    return NO_ERROR;
}

ErrorCode VulkanResize::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];

    encodeImpl(input, output, 1.0f / mXScale, 1.0f / mYScale, cmdBuffer);

    return NO_ERROR;
}

class VulkanResizeCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override {
        auto scale = op->main_as_Resize();
        return new VulkanResize(bn, scale->xScale(), scale->yScale());
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Resize, new VulkanResizeCreator);
    return true;
}();

} // namespace MNN
