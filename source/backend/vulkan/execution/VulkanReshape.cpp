//
//  VulkanReshape.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanReshape.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {

VulkanReshape::VulkanReshape(const Op* op, Backend* bn) : VulkanBasicExecution(bn), mStorage(2) {
    mDimType       = op->main_as_Reshape()->dimType();
    auto vkBackend = static_cast<VulkanBackend*>(bn);
    mTensorConvert0.reset(new VulkanImageConverter(vkBackend));
    mTensorConvert1.reset(new VulkanImageConverter(vkBackend));
}

VulkanReshape::VulkanReshape(Backend* bn) : VulkanBasicExecution(bn), mDimType(MNN_DATA_FORMAT_NCHW), mStorage(2) {
    auto vkBackend = static_cast<VulkanBackend*>(bn);
    mTensorConvert0.reset(new VulkanImageConverter(vkBackend));
    mTensorConvert1.reset(new VulkanImageConverter(vkBackend));
}

VulkanReshape::~VulkanReshape() {
}

ErrorCode VulkanReshape::setLayout(const Tensor* input, const Tensor* output) {
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
    mStorage.buffer().dim[1].flags  = 0;
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

    if (input->buffer().dimensions > 1) {
        mWrapTensorForInput.buffer().dim[1].flags = 0;
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
    if (output->buffer().dimensions > 1) {
        mWrapTensorForOutput.buffer().dim[1].flags = 0;
    }
    mWrapTensorForOutput.buffer().device = mStorage.buffer().device;
    TensorUtils::setLinearLayout(&mWrapTensorForOutput);
    return NO_ERROR;
}

ErrorCode VulkanReshape::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const VulkanCommandPool::Buffer* cmdBuffer) {
    MNN_ASSERT(1 == inputs.size() || 2 == inputs.size());
    MNN_ASSERT(1 == outputs.size());

    auto input  = inputs[0];
    auto output = outputs[0];

    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NHWC &&
        TensorUtils::getDescribe(output)->dimensionFormat == MNN_DATA_FORMAT_NHWC) {
        // the layout of input and output tensor are all NHWC, then copy buffer directly
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

class VulkanReshapeCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanReshape(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Reshape, new VulkanReshapeCreator);
    return true;
}();

} // namespace MNN
