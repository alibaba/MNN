//
//  VulkanTensorConvert.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/execution/VulkanTensorConvert.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

VulkanTensorConvertVulkanBasicExecution::VulkanTensorConvertVulkanBasicExecution(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto vkBackend = static_cast<VulkanBackend*>(bn);

    mTensorConverter = std::make_shared<VulkanImageConverter>(vkBackend);
}

VulkanTensorConvertVulkanBasicExecution::~VulkanTensorConvertVulkanBasicExecution() {
}

ErrorCode VulkanTensorConvertVulkanBasicExecution::onEncode(const std::vector<Tensor*>& inputs,
                                                 const std::vector<Tensor*>& outputs,
                                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input     = inputs[0];
    auto output    = outputs[0];
    auto vkBackend = static_cast<VulkanBackend*>(backend());

    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        auto vkTensor = vkBackend->findTensor(output->deviceId());
        MNN_ASSERT(TensorUtils::getDescribe(output)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4);
        mTensorConverter->encodeTensorToBuffer(input, vkTensor->buffer()->buffer(), vkTensor->buffer()->size(), 0,
                                               TensorUtils::getDescribe(output)->dimensionFormat, cmdBuffer);
    } else {
        MNN_ASSERT(TensorUtils::getDescribe(output)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4);
        auto vkTensor = vkBackend->findTensor(input->deviceId());
        mTensorConverter->encodeBufferToTensor(vkTensor->buffer()->buffer(), output, vkTensor->buffer()->size(), 0,
                                               TensorUtils::getDescribe(input)->dimensionFormat, cmdBuffer);
    }

    return NO_ERROR;
}

class VulkanTensorConvertVulkanBasicExecutionCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanTensorConvertVulkanBasicExecution(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_ConvertTensor, new VulkanTensorConvertVulkanBasicExecutionCreator);
    return true;
}();

} // namespace MNN
