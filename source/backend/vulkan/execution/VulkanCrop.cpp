//
//  VulkanCrop.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanCrop.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"
namespace MNN {

VulkanCrop::VulkanCrop(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto cropParam        = op->main_as_Crop();
    mAxis                 = cropParam->axis();
    const auto offsetSize = cropParam->offset()->size();
    mCropOffset.resize(offsetSize);
    ::memcpy(mCropOffset.data(), cropParam->offset()->data(), offsetSize * sizeof(int));
}
VulkanCrop::~VulkanCrop() {
}

ErrorCode VulkanCrop::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) {
    // !!! now only support crop spatial
    const auto input0 = inputs[0];
    const auto input1 = inputs[1];
    auto output       = outputs[0];
    MNN_ASSERT(TensorUtils::getDescribe(input0)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4);
    const int inputDim = input0->buffer().dimensions;
    std::vector<int> offsets(inputDim, 0);
    MNN_ASSERT(2 <= mAxis);
    for (int i = 0; i < inputDim; ++i) {
        int cropOffset = 0;
        if (i >= mAxis) {
            if (mCropOffset.size() == 1) {
                cropOffset = mCropOffset[0];
            } else {
                cropOffset = mCropOffset[i - mAxis];
            }
            MNN_ASSERT(input0->length(i) - cropOffset >= input1->length(i));
        }
        offsets[i] = cropOffset;
    }

    VkImageCopy copyRegion;
    ::memset(&copyRegion, 0, sizeof(VkImageCopy));
    copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.srcSubresource.layerCount = 1;
    copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.dstSubresource.layerCount = 1;
    copyRegion.srcOffset.x               = offsets[3]; // width offset
    copyRegion.srcOffset.y               = offsets[2]; // height offset
    copyRegion.srcOffset.z               = offsets[1]; // channels offset
    copyRegion.extent.width              = output->width();
    copyRegion.extent.height             = output->height();
    copyRegion.extent.depth              = UP_DIV(output->channel(), 4) * output->batch();

    auto vkBackend   = static_cast<VulkanBackend*>(backend());
    auto input0Image = vkBackend->findTensor(input0->deviceId())->image();
    auto outputImage = vkBackend->findTensor(output->deviceId())->image();
    vkCmdCopyImage(cmdBuffer->get(), input0Image->get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, outputImage->get(),
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

    return NO_ERROR;
}

class VulkanCropCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanCrop(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Crop, new VulkanCropCreator);
    return true;
}();

} // namespace MNN
