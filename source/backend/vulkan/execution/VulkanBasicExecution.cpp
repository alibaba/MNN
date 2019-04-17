//
//  VulkanBasicExecution.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanBasicExecution.hpp"
#include "VulkanBackend.hpp"
namespace MNN {
VulkanBasicExecution::VulkanBasicExecution(Backend *bn) : Execution(bn) {
    auto extra = static_cast<VulkanBackend *>(bn);
    mCmdBuffer.reset(const_cast<VulkanCommandPool::Buffer *>(extra->getPool().allocBuffer()));
}
VulkanBasicExecution::~VulkanBasicExecution() {
}

ErrorCode VulkanBasicExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto extra = static_cast<VulkanBackend *>(backend());
    extra->pushCommand(mCmdBuffer->get());
    return NO_ERROR;
}

ErrorCode VulkanBasicExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mCmdBuffer->begin(0);
    auto extra = static_cast<VulkanBackend *>(backend());
    for (auto input : inputs) {
        auto vkTensor = extra->findTensor(input->deviceId());
        if (nullptr == vkTensor) {
            // The case occured if we don't need the content of input
            continue;
        }
        if (nullptr != vkTensor->image()) {
            mCmdBuffer->barrierImage(vkTensor->image()->get(), VK_IMAGE_LAYOUT_GENERAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        } else {
            MNN_ASSERT(vkTensor->buffer() != nullptr);
            mCmdBuffer->barrierSource(vkTensor->buffer()->buffer(), 0, vkTensor->buffer()->size());
        }
    }
    auto code = this->onEncode(inputs, outputs, mCmdBuffer.get());
    mCmdBuffer->end();
    return code;
}

} // namespace MNN
