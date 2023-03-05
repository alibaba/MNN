//
//  VulkanBasicExecution.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanBasicExecution.hpp"
#include "VulkanBackend.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
VulkanBasicExecutionDirect::VulkanBasicExecutionDirect(std::shared_ptr<VulkanBasicExecution> encoder) : Execution(encoder->backend()) {
    mEncoder = encoder;
    auto extra = static_cast<VulkanBackend *>(encoder->backend());
    mCmdBuffer.reset(const_cast<VulkanCommandPool::Buffer *>(extra->getPool().allocBuffer()));
}

ErrorCode VulkanBasicExecutionDirect::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto extra = static_cast<VulkanBackend *>(backend());
    extra->pushCommand(mCmdBuffer->get());
    return NO_ERROR;
}

ErrorCode VulkanBasicExecutionDirect::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mCmdBuffer->begin(0);
    auto vkBn = static_cast<VulkanBackend*>(backend());
    for (auto input : inputs) {
        auto des = TensorUtils::getDescribe(input);
        if (0 == input->deviceId()) {
            continue;
        }
        auto vkTensor = (VulkanBuffer*)(input->deviceId());
        if (nullptr == vkTensor) {
            // The case occured if we don't need the content of input
            continue;
        }
        auto offset = des->extra.offset;
        mCmdBuffer->barrierSource(vkTensor->buffer(), offset, vkBn->getTensorSize(input));
    }
    auto code = mEncoder->onEncode(inputs, outputs, mCmdBuffer.get());

    mCmdBuffer->end();
    return code;
}
VulkanBasicExecutionInDirect::VulkanBasicExecutionInDirect(std::shared_ptr<VulkanBasicExecution> encoder) : Execution(encoder->backend()) {
    mEncoder = encoder;
}
ErrorCode VulkanBasicExecutionInDirect::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto extra = static_cast<VulkanBackend *>(backend());
    auto mCmdBuffer = extra->getSingleCommand();
    auto vkBn = static_cast<VulkanBackend*>(backend());
    for (auto input : inputs) {
        auto vkTensor = (VulkanBuffer*)(input->deviceId());
        auto des = TensorUtils::getDescribe(input);
        if (nullptr == vkTensor) {
            // The case occured if we don't need the content of input
            continue;
        }
        auto offset = des->extra.offset;
        mCmdBuffer->barrierSource(vkTensor->buffer(), offset, vkBn->getTensorSize(input));
    }
    auto code = mEncoder->onEncode(inputs, outputs, mCmdBuffer.get());
    return code;
}
} // namespace MNN
