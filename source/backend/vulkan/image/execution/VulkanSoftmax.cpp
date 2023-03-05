//
//  VulkanSoftmax.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanSoftmax.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

struct ConstBuffer {
    int w;
    int h;
    int c;
};

VulkanSoftmax::VulkanSoftmax(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    const auto softmaxParam = op->main_as_Axis();
    mAxis                   = softmaxParam->axis();
    auto vkBn = (VulkanBackend*)backend();
    mConstBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, sizeof(ConstBuffer), nullptr,
                                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    mSoftmaxPipeline =
        vkBn->getPipeline("glsl_softmaxHeight_NHWC_comp", types);
    mDescriptorSet.reset(mSoftmaxPipeline->createSet());
    mSource.convert.reset(new VulkanImageConverter(vkBn));
    mOutput.convert.reset(new VulkanImageConverter(vkBn));
}

VulkanSoftmax::~VulkanSoftmax() {
}

ErrorCode VulkanSoftmax::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];

    auto inputFormat = TensorUtils::getDescribe(input)->dimensionFormat;
    auto axis = mAxis;
    if (axis < 0) {
        axis = input->dimensions() + axis;
    }
    auto mVkBackend = (VulkanBackend*)backend();
    int inside = 1;
    int outside = 1;
    int mid = input->length(axis);
    for (int i=0; i<axis; ++i) {
        outside *= input->length(i);
    }
    for (int i=axis+1; i<output->dimensions(); ++i) {
        inside *= input->length(i);
    }
    // gpu param
    {
        auto softmax = reinterpret_cast<ConstBuffer*>(mConstBuffer->map());
        ::memset(softmax, 0, sizeof(ConstBuffer));
        softmax->w = inside;
        softmax->h = mid;
        softmax->c = outside;
        mConstBuffer->unmap();
    }
    auto vkBn = static_cast<VulkanBackend*>(backend());
    {
        int bufferSize = sizeof(float);
        for (int i=0; i<input->dimensions(); ++i) {
            bufferSize *= input->length(i);
        }
        mSource.buffer.reset(new VulkanBuffer(vkBn->getDynamicMemoryPool(),
                                           false, bufferSize, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    }
    {
        int bufferSize = sizeof(float);
        for (int i=0; i<output->dimensions(); ++i) {
            bufferSize *= output->length(i);
        }
        mOutput.buffer.reset(new VulkanBuffer(vkBn->getDynamicMemoryPool(), false, bufferSize, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    }

    // Encode
    mSource.convert->encodeTensorToBuffer(input, mSource.buffer->buffer(), mSource.buffer->size(), 0, VulkanImageConverter::getTensorLinearFormat(input), cmdBuffer);

    mDescriptorSet->writeBuffer(mOutput.buffer->buffer(), 0, mOutput.buffer->size());
    mDescriptorSet->writeBuffer(mSource.buffer->buffer(), 1, mSource.buffer->size());
    mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 2, mConstBuffer->size());
    cmdBuffer->barrierSource(mSource.buffer->buffer(), 0, mSource.buffer->size());
    mSoftmaxPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(outside, 8), UP_DIV(inside, 8), 1);
    cmdBuffer->barrierSource(mOutput.buffer->buffer(), 0, mOutput.buffer->size());
    mOutput.convert->encodeBufferToTensor(mOutput.buffer->buffer(), output, mOutput.buffer->size(), 0, VulkanImageConverter::getTensorLinearFormat(output), cmdBuffer);
    {
        mSource.buffer->release();
        mOutput.buffer->release();
    }
    return NO_ERROR;
}

class VulkanSoftmaxCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new VulkanSoftmax(op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Softmax, new VulkanSoftmaxCreator);
    return true;
}();

} // namespace MNN
