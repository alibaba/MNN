//
//  VulkanArgMax.cpp
//  MNN
//
//  Created by MNN on 2024/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanArgMax.hpp"

namespace MNN {

struct GpuArgMaxParam {
    ivec4 size; // inside, mid, outside, 0
};

VulkanArgMax::VulkanArgMax(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto vkBn = (VulkanBackend *)backend();

    mAxis = op->main_as_ArgMax()->axis();

    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    if (op->type() == OpType_ArgMax) {
        mArgMaxPipeline =
            vkBn->getPipeline("glsl_argmax_comp", types);
    } else {
        MNN_ASSERT(op->type() == OpType_ArgMin);
        mArgMaxPipeline =
            vkBn->getPipeline("glsl_argmax_ARGMIN_comp", types);
    }

    mGpuArgMaxParam.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, sizeof(GpuArgMaxParam), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    mDescriptorSet.reset(mArgMaxPipeline->createSet());
}

VulkanArgMax::~VulkanArgMax() {
}

// set descriptorSet， including output, input and GPU param
ErrorCode VulkanArgMax::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) {
    auto vkBn = (VulkanBackend*)backend();
    auto input  = inputs[0];
    auto output = outputs[0];

    // set GPU param
    auto axis = mAxis;
    if (axis < 0) {
        axis = input->dimensions() + axis;
    }
    int inside = 1;
    int outside = 1;
    int mid = input->length(axis);
    for (int i=0; i<axis; ++i) {
        outside *= input->length(i);
    }
    for (int i=axis+1; i<input->dimensions(); ++i) {
        inside *= input->length(i);
    }
    auto total = outside * inside;

    auto Argmax = reinterpret_cast<GpuArgMaxParam *>(mGpuArgMaxParam->map());
    Argmax->size[0] = inside;
    Argmax->size[1] = mid;
    Argmax->size[2] = outside;
    Argmax->size[3] = 0;
    mGpuArgMaxParam->unmap();

    // set necessary storages, set descriptorSet and bind commandBuffer
    {
        int bufferSizeSource = sizeof(float);
        for (int i=0; i<input->dimensions(); ++i) {
            bufferSizeSource *= input->length(i);
        }
        mSource.buffer.reset(new VulkanBuffer(vkBn->getDynamicMemoryPool(), false, bufferSizeSource, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        mSource.convert.reset(new VulkanImageConverter(vkBn));
    }
    {
        int bufferSizeOutput = sizeof(float);
        for (int i=0; i<output->dimensions(); ++i) {
            bufferSizeOutput *= output->length(i);
        }
        mOutput.convert.reset(new VulkanImageConverter(vkBn));
        mOutput.buffer.reset(new VulkanBuffer(vkBn->getDynamicMemoryPool(), false, bufferSizeOutput, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    }

    mSource.convert->encodeTensorToBuffer(input, mSource.buffer->buffer(), mSource.buffer->size(), 0, VulkanImageConverter::getTensorLinearFormat(input), cmdBuffer);

    mDescriptorSet->writeBuffer(mOutput.buffer->buffer(), 0, mOutput.buffer->size());
    mDescriptorSet->writeBuffer(mSource.buffer->buffer(), 1, mSource.buffer->size());
    mDescriptorSet->writeBuffer(mGpuArgMaxParam->buffer(), 2, mGpuArgMaxParam->size());

    cmdBuffer->barrierSource(mSource.buffer->buffer(), 0, mSource.buffer->size());

    mArgMaxPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(total, 256), 1, 1);

    cmdBuffer->barrierSource(mOutput.buffer->buffer(), 0, mOutput.buffer->size());
    mOutput.convert->encodeBufferToTensor(mOutput.buffer->buffer(), output, mOutput.buffer->size(), 0, VulkanImageConverter::getTensorLinearFormat(output), cmdBuffer);
    {
        mSource.buffer->release();
        mOutput.buffer->release();
    }
    return NO_ERROR;
}

class VulkanArgMaxCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            // Don't support legency version
            return nullptr;
        }
        return new VulkanArgMax(op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_ArgMax, new VulkanArgMaxCreator);
    VulkanBackend::addCreator(OpType_ArgMin, new VulkanArgMaxCreator);
    return true;
}();

}
