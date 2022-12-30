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
//#define MNN_VULKAN_DEBUG
//#define MNN_VULKAN_DEBUG_EAGER
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

static void _initLayout(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const VulkanCommandPool::Buffer* initCmdBuffer) {
    for (auto input : inputs) {
        auto vkTensor = reinterpret_cast<VulkanTensor*>(input->deviceId());
        if (nullptr == vkTensor) {
            continue;
        }
        for (int i=0; i<vkTensor->imageSize(); ++i) {
            auto img = vkTensor->image(i);
            if (img->currentLayout() == VK_IMAGE_LAYOUT_UNDEFINED) {
                img->barrierRead(initCmdBuffer->get());
            }
        }
    }
}
static void _postTreat(const std::vector<Tensor *> &outputs, const VulkanCommandPool::Buffer* initCmdBuffer) {
    for (auto output : outputs) {
        auto vkTensor = reinterpret_cast<VulkanTensor*>(output->deviceId());
        if (nullptr == vkTensor) {
            continue;
        }
        for (int i=0; i<vkTensor->imageSize(); ++i) {
            auto img = vkTensor->image(i);
            if (img->currentLayout() == VK_IMAGE_LAYOUT_UNDEFINED) {
                auto img = vkTensor->image(i);
                img->barrierRead(initCmdBuffer->get());
            }
        }
    }
}

ErrorCode VulkanBasicExecutionDirect::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto initCmdBuffer = static_cast<VulkanBackend*>(backend())->getInitCommandBuffer();
    _initLayout(inputs, outputs, initCmdBuffer);
    mCmdBuffer->begin(0);
    auto code = mEncoder->onEncode(inputs, outputs, mCmdBuffer.get());
    for (auto output : outputs) {
        auto vkTensor = reinterpret_cast<VulkanTensor*>(output->deviceId());
        for (int i=0; i<vkTensor->imageSize(); ++i) {
            auto img = vkTensor->image(i);
            img->barrierRead(mCmdBuffer->get());
        }
    }
    _postTreat(outputs, mCmdBuffer.get());
    mCmdBuffer->end();
#ifdef MNN_VULKAN_DEBUG
#ifdef MNN_VULKAN_DEBUG_EAGER
    static_cast<VulkanBackend*>(backend())->onExecuteBegin();
    static_cast<VulkanBackend*>(backend())->pushCommand(mCmdBuffer->get());
    static_cast<VulkanBackend*>(backend())->onExecuteEnd();
#endif
#endif
    return code;
}
VulkanBasicExecutionInDirect::VulkanBasicExecutionInDirect(std::shared_ptr<VulkanBasicExecution> encoder) : Execution(encoder->backend()) {
    mEncoder = encoder;
}
ErrorCode VulkanBasicExecutionInDirect::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto extra = static_cast<VulkanBackend *>(backend());
    auto initCmdBuffer = static_cast<VulkanBackend*>(backend())->getInitCommandBuffer();
    _initLayout(inputs, outputs, initCmdBuffer);
    auto mCmdBuffer = extra->getSingleCommand();
    auto code = mEncoder->onEncode(inputs, outputs, mCmdBuffer.get());
    _postTreat(outputs, mCmdBuffer.get());
    return code;
}
} // namespace MNN
