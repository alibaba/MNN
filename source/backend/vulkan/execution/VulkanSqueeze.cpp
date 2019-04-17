//
//  VulkanSqueeze.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanSqueeze.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"
namespace MNN {
VulkanSqueeze::VulkanSqueeze(Backend* bn) : VulkanBasicExecution(bn) {
}

VulkanSqueeze::~VulkanSqueeze() {
}

ErrorCode VulkanSqueeze::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input       = inputs[0];
    auto output      = outputs[0];
    auto inputFormat = TensorUtils::getDescribe(input)->dimensionFormat;
    if (MNN_DATA_FORMAT_NC4HW4 == inputFormat) {
        VkImageCopy copyRegion;
        ::memset(&copyRegion, 0, sizeof(copyRegion));
        copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.srcSubresource.layerCount = 1;
        copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.dstSubresource.layerCount = 1;
        copyRegion.extent.width              = input->width();
        copyRegion.extent.height             = input->height();
        copyRegion.extent.depth              = UP_DIV(input->channel(), 4);
        auto vkBackend                       = static_cast<VulkanBackend*>(backend());
        auto inputImage                      = vkBackend->findTensor(input->deviceId())->image();
        auto outputImage                     = vkBackend->findTensor(output->deviceId())->image();
        vkCmdCopyImage(cmdBuffer->get(), inputImage->get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, outputImage->get(),
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
    } else {
        auto inputBuffer  = reinterpret_cast<VkBuffer>(input->deviceId());
        auto outputBuffer = reinterpret_cast<VkBuffer>(output->deviceId());
        cmdBuffer->barrierSource(inputBuffer, 0, input->size());
        const VkBufferCopy copyRegion = {0, 0, static_cast<VkDeviceSize>(input->size())};
        vkCmdCopyBuffer(cmdBuffer->get(), inputBuffer, outputBuffer, 1, &copyRegion);
    }
    return NO_ERROR;
}

class VulkanSqueezeCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanSqueeze(bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Squeeze, new VulkanSqueezeCreator);
    return true;
}();
} // namespace MNN
