//
//  VulkanBatchToSpaceND.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanBatchToSpaceND.hpp"

#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {

struct GpuParamSpaceBatch {
    ivec4 inImageSize;
    ivec4 outImageSize;
    ivec2 crops;
    ivec2 blockShape;
};

VulkanBatchToSpaceND::VulkanBatchToSpaceND(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto param        = op->main_as_SpaceBatch();
    mCropTop          = param->padding()->int32s()->data()[0];
    mCropLeft         = param->padding()->int32s()->data()[1];
    mBlockShapeHeight = param->blockShape()->int32s()->data()[0];
    mBlockShapeWidth  = param->blockShape()->int32s()->data()[1];

    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };

    auto vkBackend        = static_cast<VulkanBackend*>(bn);
    mBatchToSpacePipeline = vkBackend->getPipeline("glsl_BatchToSpaceND_comp", types);
    mGpuParam = std::make_shared<VulkanBuffer>(vkBackend->getMemoryPool(), false, sizeof(GpuParamSpaceBatch), nullptr,
                                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    mSampler  = vkBackend->getCommonSampler();
}

VulkanBatchToSpaceND::~VulkanBatchToSpaceND() {
}

ErrorCode VulkanBatchToSpaceND::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                         const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];
    MNN_ASSERT(MNN_DATA_FORMAT_NC4HW4 == TensorUtils::getDescribe(input)->dimensionFormat);

    auto param             = reinterpret_cast<GpuParamSpaceBatch*>(mGpuParam->map());
    param->inImageSize[0]  = input->width();
    param->inImageSize[1]  = input->height();
    param->inImageSize[2]  = UP_DIV(input->channel(), 4);
    param->inImageSize[3]  = input->channel();
    param->outImageSize[0] = output->width();
    param->outImageSize[1] = output->height();
    param->outImageSize[2] = UP_DIV(output->channel(), 4);
    param->outImageSize[3] = output->batch();
    param->crops[0]        = mCropTop;
    param->crops[1]        = mCropLeft;
    param->blockShape[0]   = mBlockShapeHeight;
    param->blockShape[1]   = mBlockShapeWidth;
    mGpuParam->unmap();

    mDescriptorSet.reset(mBatchToSpacePipeline->createSet());
    mDescriptorSet->writeImage(reinterpret_cast<VkImageView>(output->deviceId()), mSampler->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 0);
    mDescriptorSet->writeImage(reinterpret_cast<VkImageView>(input->deviceId()), mSampler->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mDescriptorSet->writeBuffer(mGpuParam->buffer(), 2, mGpuParam->size());
    mBatchToSpacePipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(input->width(), 16), UP_DIV(input->height(), 16),
                  UP_DIV(input->channel(), 4) * input->batch());

    return NO_ERROR;
}

class VulkanBatchToSpaceNDCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanBatchToSpaceND(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_BatchToSpaceND, new VulkanBatchToSpaceNDCreator);
    return true;
}();

} // namespace MNN
