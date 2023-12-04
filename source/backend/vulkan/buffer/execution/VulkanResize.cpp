//
//  VulkanResize.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanResize.hpp"
#include "core/Macro.h"

namespace MNN {
struct GpuParam {
    ivec4 inImgSize;
    ivec4 outImgSize;
    vec4 cord;
};

VulkanResize::VulkanResize(Backend* bn, float xScale, float yScale, int resizeType)
    : VulkanBasicExecution(bn), mXScale(xScale), mYScale(yScale) {
    std::vector<VkDescriptorType> VulkanResizeTypes{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    auto extra            = static_cast<VulkanBackend*>(bn);
    if (1 == resizeType) {
        mVulkanResizePipeline = extra->getPipeline(
                                                   "glsl_resizeNearest_comp", VulkanResizeTypes);
    } else if (2 == resizeType) {
        mVulkanResizePipeline = extra->getPipeline(
                                                   "glsl_resizeBilinear_comp", VulkanResizeTypes);
    } else {
        mVulkanResizePipeline = extra->getPipeline(
                                                   "glsl_resizeNearest_NEAREST_ROUND_comp", VulkanResizeTypes);
    }
    mParamBuffer.reset(
        new VulkanBuffer(extra->getMemoryPool(), false, sizeof(GpuParam), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    mDescriptorSet.reset(mVulkanResizePipeline->createSet());
}
VulkanResize::~VulkanResize() {
}

ErrorCode VulkanResize::encodeImpl(Tensor* input, Tensor* output, const float* cords,
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
    ::memcpy(VulkanResizeParam->cord, cords, 4 * sizeof(float));
    mParamBuffer->unmap();

    auto vkOutput = extra->getBuffer(output);
    auto vkInput  = extra->getBuffer(input);
    
    mDescriptorSet->writeBuffer(vkOutput, 0);
    mDescriptorSet->writeBuffer(vkInput, 1);
    mDescriptorSet->writeBuffer(mParamBuffer->buffer(), 2, mParamBuffer->size());
    mVulkanResizePipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(output->width(), 16), UP_DIV(output->height(), 16),
                  channelDiv4 * input->batch());

    return NO_ERROR;
}


} // namespace MNN
