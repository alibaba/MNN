//
//  VulkanSlice.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanSlice.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {

VulkanSlice::VulkanSlice(const Op* op, Backend* backend) : VulkanBasicExecution(backend), mTempTensor(4) {
    const auto sliceParam = op->main_as_Slice();
    mAxis                 = sliceParam->axis();
    // [TODO] now only support slice on channel
    MNN_ASSERT(1 == mAxis);
    auto vkBackend         = static_cast<VulkanBackend*>(backend);
    mTensorConverter4Input = std::make_shared<VulkanImageConverter>(vkBackend);
    for (int i = 0; i < sliceParam->slicePoints()->size() + 1; ++i) {
        auto tensorConvert = std::make_shared<VulkanImageConverter>(vkBackend);
        mTensorConverters4Ouput.push_back(tensorConvert);
    }
}

VulkanSlice::~VulkanSlice() {
}

ErrorCode VulkanSlice::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input                 = inputs[0];
    auto inputDim              = input->buffer().dim;
    const int height           = std::max(inputDim[2].extent, 1);
    const int width            = std::max(inputDim[3].extent, 1);
    const int inputPlaneStride = height * width;
    bool channelAligned        = true;
    for (size_t n = 0; n < outputs.size(); ++n) {
        auto& ob = outputs[n]->buffer();
        if (ob.dim[1].extent % 4 != 0) {
            channelAligned = false;
            break;
        }
    }
    MNN_ASSERT(TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4);
    if (channelAligned) {
        VkImageCopy copyRegion;
        ::memset(&copyRegion, 0, sizeof(copyRegion));
        copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.srcSubresource.layerCount = 1;
        copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.dstSubresource.layerCount = 1;
        copyRegion.extent.width              = input->width();
        copyRegion.extent.height             = input->height();
        auto vkbackend                       = static_cast<VulkanBackend*>(backend());
        auto inputImage                      = vkbackend->findTensor(input->deviceId())->image()->get();
        int currentChannels                  = 0;
        for (size_t ni = 0; ni < outputs.size(); ++ni) {
            auto curOutput          = outputs[ni];
            auto curOutputImage     = vkbackend->findTensor(curOutput->deviceId())->image()->get();
            copyRegion.srcOffset.z  = UP_DIV(currentChannels, 4);
            copyRegion.extent.depth = UP_DIV(curOutput->channel(), 4);
            vkCmdCopyImage(cmdBuffer->get(), inputImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, curOutputImage,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
            currentChannels += curOutput->channel();
        }
    } else {
        // set mTempTensor's layout to be NCHW
        mTempTensor.buffer().type = input->buffer().type;
        TensorUtils::copyShape(input, &mTempTensor);
        TensorUtils::getDescribe(&mTempTensor)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        mTempTensor.buffer().dim[1].flags                       = 0;
        backend()->onAcquireBuffer(&mTempTensor, Backend::DYNAMIC);

        mTensorConverter4Input->encodeTensorToBuffer(input, reinterpret_cast<VkBuffer>(mTempTensor.deviceId()),
                                                     mTempTensor.size(), 0, MNN_DATA_FORMAT_NCHW, cmdBuffer);
        cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(mTempTensor.deviceId()), 0, mTempTensor.size());
        int currentChannels = 0;
        for (size_t ni = 0; ni < outputs.size(); ++ni) {
            auto curOutput            = outputs[ni];
            const int curBufferSize   = curOutput->channel() * inputPlaneStride * sizeof(float);
            const VkDeviceSize offset = currentChannels * inputPlaneStride * sizeof(float);
            mTensorConverters4Ouput[ni]->encodeBufferToTensor(reinterpret_cast<VkBuffer>(mTempTensor.deviceId()),
                                                              curOutput, curBufferSize, offset, MNN_DATA_FORMAT_NCHW,
                                                              cmdBuffer);
            currentChannels += curOutput->channel();
        }

        backend()->onReleaseBuffer(&mTempTensor, Backend::DYNAMIC);
    }

    return NO_ERROR;
}

class VulkanSliceCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override {
        const auto sliceParam = op->main_as_Slice();
        if (1 != sliceParam->axis()) {
            MNN_PRINT("Vulkan slice don't support %d axis slice\n", sliceParam->axis());
            return nullptr;
        }
        return new VulkanSlice(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Slice, new VulkanSliceCreator);
    return true;
}();
} // namespace MNN
