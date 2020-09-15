//
//  VulkanPadding.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/execution/VulkanPadding.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

VulkanPadding::VulkanPadding(PadValueMode mode, int32_t* paddings, Backend* bn) : VulkanBasicExecution(bn), mMode(mode) {
    ::memcpy(mPaddings, paddings, sizeof(int32_t) * 8);
    mDimType = MNN_DATA_FORMAT_NCHW;
}

VulkanPadding::~VulkanPadding() {
}

ErrorCode VulkanPadding::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const VulkanCommandPool::Buffer* cmdBuffer) {
    MNN_ASSERT(1 <= inputs.size());
    MNN_ASSERT(1 == outputs.size());

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto vkBackend   = static_cast<VulkanBackend*>(backend());
    auto inputImage  = vkBackend->findTensor(input->deviceId())->image();
    auto outputImage = vkBackend->findTensor(output->deviceId())->image();

    cmdBuffer->barrierImageIfNeeded(inputImage, VK_IMAGE_LAYOUT_GENERAL);
    cmdBuffer->barrierImageIfNeeded(outputImage, VK_IMAGE_LAYOUT_GENERAL);

    VkClearColorValue colorValue;
    VkImageSubresourceRange range;
    colorValue.float32[0] = 0.0f;
    colorValue.float32[1] = 0.0f;
    colorValue.float32[2] = 0.0f;
    colorValue.float32[3] = 0.0f;
    range.levelCount      = 1;
    range.layerCount      = 1;
    range.baseMipLevel    = 0;
    range.baseArrayLayer  = 0;
    range.aspectMask      = VK_IMAGE_ASPECT_COLOR_BIT;
    vkCmdClearColorImage(cmdBuffer->get(), outputImage->get(), outputImage->layout(),
                         &colorValue, 1, &range);

    VkImageCopy copyRegion;
    ::memset(&copyRegion, 0, sizeof(VkImageCopy));
    copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.srcSubresource.layerCount = 1;
    copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.dstSubresource.layerCount = 1;
    copyRegion.srcOffset.x               = mPaddings[4]; // width offset
    copyRegion.srcOffset.y               = mPaddings[6]; // height offset
    copyRegion.srcOffset.z               = mPaddings[2]; // channels offset
    copyRegion.extent.width              = input->width();
    copyRegion.extent.height             = input->height();
    copyRegion.extent.depth              = UP_DIV(input->channel(), 4) * input->batch();
    vkCmdCopyImage(cmdBuffer->get(), inputImage->get(), inputImage->layout(),
                   outputImage->get(), outputImage->layout(), 1, &copyRegion);

    return NO_ERROR;
}

class VulkanPaddingCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        if (inputs.size() < 2) {
            MNN_ERROR("Need second input for padding parameters\n");
            return nullptr;
        }
        auto padding = inputs[1]->host<int32_t>();
        auto paddingShape = inputs[1]->shape();
        int paddingSize = 1;
        for (auto dim: paddingShape)
            paddingSize *= dim;
        if (paddingSize != 8) {
            MNN_ERROR("Padding parameter size should be 8 for [NCHW min][NCHW max]\n");
            return nullptr;
        }
        auto param = op->main_as_PadParam();
        auto mode  = PadValueMode_CONSTANT;
        if (param) {
            mode = param->mode();
        }

        return new VulkanPadding(mode, padding, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Padding, new VulkanPaddingCreator);
    return true;
}();

} // namespace MNN
