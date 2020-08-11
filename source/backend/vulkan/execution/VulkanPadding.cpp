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

VulkanPadding::VulkanPadding(PadValueMode mode, int32_t* paddings, Backend* bn) : VulkanBasicExecution(bn), mMode(mode), mStorage(2) {
    ::memcpy(mPaddings, paddings, sizeof(int32_t) * 8);
    mDimType = MNN_DATA_FORMAT_NCHW;
    auto vkBackend = static_cast<VulkanBackend*>(bn);
    mTensorConvert0.reset(new VulkanImageConverter(vkBackend));
    mTensorConvert1.reset(new VulkanImageConverter(vkBackend));
}

VulkanPadding::~VulkanPadding() {
}

ErrorCode VulkanPadding::setLayout(const Tensor* input, const Tensor* output) {
    int totalSize = 1;

    mWrapTensorForInput.buffer().type  = input->buffer().type;
    mWrapTensorForOutput.buffer().type = output->buffer().type;
    int extraMulti                     = 1;
    int extraDivide                    = 1;
    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        TensorUtils::getDescribe(&mWrapTensorForInput)->dimensionFormat  = MNN_DATA_FORMAT_NCHW;
        TensorUtils::getDescribe(&mWrapTensorForOutput)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        extraMulti                                                       = ALIGN_UP4(input->channel());
        extraDivide                                                      = input->channel();
    } else {
        TensorUtils::getDescribe(&mWrapTensorForInput)->dimensionFormat  = MNN_DATA_FORMAT_NHWC;
        TensorUtils::getDescribe(&mWrapTensorForOutput)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
    }

    for (int i = 0; i < input->buffer().dimensions; ++i) {
        totalSize *= input->buffer().dim[i].extent;
    }

    mStorage.buffer().dim[0].extent = 1;
    mStorage.buffer().dim[1].extent = totalSize / extraDivide * extraMulti;
    backend()->onAcquireBuffer(&mStorage, Backend::DYNAMIC);

    TensorUtils::copyShape(input, &mWrapTensorForInput);
    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
        mDimType == MNN_DATA_FORMAT_NHWC) {
        TensorUtils::getDescribe(&mWrapTensorForInput)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        if (mWrapTensorForInput.buffer().dimensions == 4) {
            mWrapTensorForInput.buffer().dim[1].extent = mWrapTensorForInput.buffer().dim[2].extent;
            mWrapTensorForInput.buffer().dim[2].extent = mWrapTensorForInput.buffer().dim[3].extent;
            mWrapTensorForInput.buffer().dim[3].extent = mWrapTensorForInput.buffer().dim[1].extent;
        }
    }

    mWrapTensorForInput.buffer().device = mStorage.buffer().device;
    TensorUtils::setLinearLayout(&mWrapTensorForInput);

    TensorUtils::copyShape(output, &mWrapTensorForOutput);
    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
        mDimType == MNN_DATA_FORMAT_NHWC) {
        TensorUtils::getDescribe(&mWrapTensorForOutput)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        if (mWrapTensorForOutput.buffer().dimensions == 4) {
            mWrapTensorForOutput.buffer().dim[1].extent = mWrapTensorForOutput.buffer().dim[2].extent;
            mWrapTensorForOutput.buffer().dim[2].extent = mWrapTensorForOutput.buffer().dim[3].extent;
            mWrapTensorForOutput.buffer().dim[3].extent = mWrapTensorForOutput.buffer().dim[1].extent;
        }
    }
    mWrapTensorForOutput.buffer().device = mStorage.buffer().device;
    TensorUtils::setLinearLayout(&mWrapTensorForOutput);
    return NO_ERROR;
}

ErrorCode VulkanPadding::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const VulkanCommandPool::Buffer* cmdBuffer) {
    MNN_ASSERT(1 <= inputs.size());
    MNN_ASSERT(1 == outputs.size());

    auto input   = inputs[0];
    auto padding = inputs[1];
    auto output  = outputs[0];

    if (TensorUtils::getDescribe(input)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 &&
        TensorUtils::getDescribe(output)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
        // the layout of input and output tensor are all buffer, then copy buffer directly
        auto inputBuffer  = reinterpret_cast<VkBuffer>(input->deviceId());
        auto outputBuffer = reinterpret_cast<VkBuffer>(output->deviceId());
        cmdBuffer->barrierSource(inputBuffer, 0, input->size());
        const VkBufferCopy copyRegion = {0, 0, static_cast<VkDeviceSize>(input->size())};
        vkCmdCopyBuffer(cmdBuffer->get(), inputBuffer, outputBuffer, 1, &copyRegion);
    } else {
        this->setLayout(input, output);

        // encode tensor convert
        mTensorConvert0->encodeTensorToBuffer(
            input, reinterpret_cast<VkBuffer>(mWrapTensorForInput.deviceId()), mStorage.size(), 0,
            TensorUtils::getDescribe(&mWrapTensorForInput)->dimensionFormat, cmdBuffer);
        cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(mWrapTensorForInput.deviceId()), 0,
                                 mWrapTensorForInput.size());
        mTensorConvert1->encodeBufferToTensor(
            reinterpret_cast<VkBuffer>(mWrapTensorForOutput.deviceId()), output, mStorage.size(), 0,
            TensorUtils::getDescribe(&mWrapTensorForOutput)->dimensionFormat, cmdBuffer);

        backend()->onReleaseBuffer(&mStorage, Backend::DYNAMIC);
    }

    return NO_ERROR;
}

// class VulkanPaddingCreator : public VulkanBackend::Creator {
// public:
//     virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
//         if (inputs.size() < 2) {
//             MNN_ERROR("Need second input for padding parameters\n");
//             return nullptr;
//         }
//         auto padding = inputs[1]->host<int32_t>();
//         if (inputs[1]->size() != 8) {
//             MNN_ERROR("Padding parameter size should be 8 for [NCHW min][NCHW max]\n");
//             return nullptr;
//         }
//         auto param = op->main_as_PadParam();
//         auto mode  = PadValueMode_CONSTANT;
//         if (param) {
//             mode = param->mode();
//         }

//         return new VulkanPadding(mode, padding, bn);
//     }
// };

// static bool gResistor = []() {
//     VulkanBackend::addCreator(OpType_Padding, new VulkanPaddingCreator);
//     return true;
// }();

} // namespace MNN
