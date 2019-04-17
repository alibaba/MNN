//
//  VulkanSpaceToBatchND.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanSpaceToBatchND.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {

struct GpuParamSpaceBatch {
    ivec4 inImageSize;
    ivec4 outImageSize;
    ivec2 padding;
    ivec2 blockShape;
};

VulkanSpaceToBatchND::VulkanSpaceToBatchND(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto param        = op->main_as_SpaceBatch();
    mPadTop           = param->padding()->int32s()->data()[0];
    mPadLeft          = param->padding()->int32s()->data()[1];
    mBlockShapeHeight = param->blockShape()->int32s()->data()[0];
    mBlockShapeWidth  = param->blockShape()->int32s()->data()[1];

    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };

    auto vkBackend        = static_cast<VulkanBackend*>(bn);
    mSpaceToBatchPipeline = vkBackend->getPipeline("glsl_SpaceToBatchND_comp",
                                                   /*glsl_SpaceToBatchND_comp, glsl_SpaceToBatchND_comp_len,*/ types);
    mGpuParam = std::make_shared<VulkanBuffer>(vkBackend->getMemoryPool(), false, sizeof(GpuParamSpaceBatch), nullptr,
                                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    mSampler  = vkBackend->getCommonSampler();
}

VulkanSpaceToBatchND::~VulkanSpaceToBatchND() {
}

ErrorCode VulkanSpaceToBatchND::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                         const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];
    MNN_ASSERT(MNN_DATA_FORMAT_NC4HW4 == TensorUtils::getDescribe(input)->dimensionFormat);

    auto param             = reinterpret_cast<GpuParamSpaceBatch*>(mGpuParam->map());
    param->inImageSize[0]  = input->width();
    param->inImageSize[1]  = input->height();
    param->inImageSize[2]  = UP_DIV(input->channel(), 4);
    param->inImageSize[3]  = input->batch();
    param->outImageSize[0] = output->width();
    param->outImageSize[1] = output->height();
    param->outImageSize[2] = UP_DIV(output->channel(), 4);
    param->outImageSize[3] = output->batch();
    param->padding[0]      = mPadTop;
    param->padding[1]      = mPadLeft;
    param->blockShape[0]   = mBlockShapeHeight;
    param->blockShape[1]   = mBlockShapeWidth;
    mGpuParam->unmap();

    mDescriptorSet.reset(mSpaceToBatchPipeline->createSet());
    mDescriptorSet->writeImage(reinterpret_cast<VkImageView>(output->deviceId()), mSampler->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 0);
    mDescriptorSet->writeImage(reinterpret_cast<VkImageView>(input->deviceId()), mSampler->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mDescriptorSet->writeBuffer(mGpuParam->buffer(), 2, mGpuParam->size());
    mSpaceToBatchPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(output->width(), 16), UP_DIV(output->height(), 16),
                  UP_DIV(output->channel(), 4) * output->batch());

    return NO_ERROR;
}

class VulkanSpaceToBatchNDCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanSpaceToBatchND(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_SpaceToBatchND, new VulkanSpaceToBatchNDCreator);
    return true;
}();

} // namespace MNN
