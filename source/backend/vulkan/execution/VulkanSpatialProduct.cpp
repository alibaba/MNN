//
//  VulkanSpatialProduct.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanSpatialProduct.hpp"
#include "Macro.h"

namespace MNN {
struct GpuParam {
    ivec4 imgsize;
};

VulkanSpatialProduct::VulkanSpatialProduct(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    std::vector<VkDescriptorType> VulkanSpatialProductTypes{
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    auto extra = static_cast<VulkanBackend*>(bn);
    mVulkanSpatialProductPipeline =
        extra->getPipeline("glsl_SpatialProduct_comp",
                           /*glsl_SpatialProduct_comp, glsl_SpatialProduct_comp_len,*/ VulkanSpatialProductTypes);
    mParamBuffer = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false, sizeof(GpuParam), nullptr,
                                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    mSampler     = extra->getCommonSampler();
}
VulkanSpatialProduct::~VulkanSpatialProduct() {
}
ErrorCode VulkanSpatialProduct::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                         const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input0 = inputs[0];
    auto input1 = inputs[1];
    MNN_ASSERT(input1->channel() == 1);
    MNN_ASSERT(input0->width() == input1->width());
    MNN_ASSERT(input0->height() == input1->height());
    auto output                    = outputs[0];
    const int channelDiv4          = UP_DIV(input0->channel(), 4);
    auto VulkanSpatialProductParam = reinterpret_cast<GpuParam*>(mParamBuffer->map());
    ::memset(VulkanSpatialProductParam, 0, sizeof(GpuParam));
    VulkanSpatialProductParam->imgsize[0] = input0->width();
    VulkanSpatialProductParam->imgsize[1] = input0->height();
    VulkanSpatialProductParam->imgsize[2] = channelDiv4;
    VulkanSpatialProductParam->imgsize[3] = 0;
    mParamBuffer->flush(true, 0, sizeof(GpuParam));
    mParamBuffer->unmap();

    mDescriptorSet.reset(mVulkanSpatialProductPipeline->createSet());

    mDescriptorSet->writeImage(reinterpret_cast<VkImageView>(output->deviceId()), mSampler->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 0);
    mDescriptorSet->writeImage(reinterpret_cast<VkImageView>(input0->deviceId()), mSampler->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mDescriptorSet->writeImage(reinterpret_cast<VkImageView>(input1->deviceId()), mSampler->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
    mDescriptorSet->writeBuffer(mParamBuffer->buffer(), 3, mParamBuffer->size());

    mVulkanSpatialProductPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(input0->deviceId()), 0, input0->size());
    cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(input1->deviceId()), 0, input1->size());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(input0->width(), 8), UP_DIV(input0->height(), 8),
                  channelDiv4 * input0->batch());

    return NO_ERROR;
}

class VulkanSpatialProductCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanSpatialProduct(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_SpatialProduct, new VulkanSpatialProductCreator);
    return true;
}();

} // namespace MNN
