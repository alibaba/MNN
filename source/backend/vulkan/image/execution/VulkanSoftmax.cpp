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

struct SoftmaxConstBuffer {
    uint32_t N;
    uint32_t H;
    uint32_t W;
    uint32_t C4;
    uint32_t CLeft;
};

struct BufferSoftmaxConst {
    int inside;
    int axis;
    int outside;
};

VulkanSoftmax::VulkanSoftmax(const Op* op, Backend* bn, const uint32_t axisIndex, bool useBuffer)
    : VulkanBasicExecution(bn) {
    mAxisIndex = axisIndex;
    mAxis = op->main_as_Axis()->axis();
    mUseBuffer = useBuffer;
    auto vkBn = static_cast<VulkanBackend*>(backend());
    if (mUseBuffer) {
        mSoftmaxConstBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, sizeof(BufferSoftmaxConst),
                                                             nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        mSoftmaxPipeline = vkBn->getPipeline("glsl_softmaxHeight_NHWC_comp", types);
        mDescriptorSet.reset(mSoftmaxPipeline->createSet());
        mSource.convert.reset(new VulkanImageConverter(vkBn));
        mOutput.convert.reset(new VulkanImageConverter(vkBn));
        return;
    }

    std::string shaderName = "glsl_softmaxImage_";
    std::string macro = "";
    std::string suffix = "comp";
    switch (axisIndex) {
        case 0:
            macro = "AXIS_N_"; break;
        case 1:
            macro = "AXIS_H_"; break;
        case 2:
            macro = "AXIS_W_"; break;
        case 3:
            macro = "AXIS_C_"; break;
    }

    std::vector<VkDescriptorType> types {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};

    mSoftmaxPipeline = vkBn->getPipeline(shaderName + macro + suffix, types);
    mDescriptorSet.reset(mSoftmaxPipeline->createSet());
    mSoftmaxConstBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, sizeof(SoftmaxConstBuffer), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
}

VulkanSoftmax::~VulkanSoftmax() {
}

ErrorCode VulkanSoftmax::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const VulkanCommandPool::Buffer* cmdBuffer) {
    if (mUseBuffer) {
        return onEncodeBuffer(inputs, outputs, cmdBuffer);
    }
    return onEncodeImage(inputs, outputs, cmdBuffer);
}

ErrorCode VulkanSoftmax::onEncodeImage(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                       const VulkanCommandPool::Buffer* cmdBuffer) {
    auto vkBn = static_cast<VulkanBackend *>(backend());
    auto input  = inputs[0];
    auto output = outputs[0];
    auto inputShapeNHWC = VulkanTensor::tensorShapeFormat(input);
    std::vector<uint32_t> cpuSoftmaxConstBuffer = {(uint32_t)inputShapeNHWC[0], (uint32_t)inputShapeNHWC[1], (uint32_t)inputShapeNHWC[2], (uint32_t)UP_DIV(inputShapeNHWC[3], 4), (uint32_t)ROUND_UP(inputShapeNHWC[3], 4) - inputShapeNHWC[3]};

    {
        auto softmaxConst = reinterpret_cast<SoftmaxConstBuffer*>(mSoftmaxConstBuffer->map());
        ::memset(softmaxConst, 0, sizeof(SoftmaxConstBuffer));
        softmaxConst->N = cpuSoftmaxConstBuffer[0];
        softmaxConst->H = cpuSoftmaxConstBuffer[1];
        softmaxConst->W = cpuSoftmaxConstBuffer[2];
        softmaxConst->C4 = cpuSoftmaxConstBuffer[3];
        softmaxConst->CLeft = cpuSoftmaxConstBuffer[4];
        mSoftmaxConstBuffer->unmap();
    }

    // N * H * W * C4
    uint32_t numTotal = cpuSoftmaxConstBuffer[0] * cpuSoftmaxConstBuffer[1] * cpuSoftmaxConstBuffer[2] * cpuSoftmaxConstBuffer[3];
    uint32_t numY = numTotal / cpuSoftmaxConstBuffer[mAxisIndex];

    auto vkOutput  = (VulkanTensor*)output->deviceId();
    auto vkInput   = (VulkanTensor*)input->deviceId();

    mDescriptorSet.reset(mSoftmaxPipeline->createSet());
    mDescriptorSet->writeImage(vkOutput->image()->view(), vkBn->getCommonSampler()->get(),
                            VK_IMAGE_LAYOUT_GENERAL, 0);
    mDescriptorSet->writeImage(vkInput->image()->view(), vkBn->getCommonSampler()->get(),
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mDescriptorSet->writeBuffer(mSoftmaxConstBuffer->buffer(), 2, mSoftmaxConstBuffer->size());

    vkOutput->image()->barrierWrite(cmdBuffer->get());
    vkInput->image()->barrierRead(cmdBuffer->get());

    mSoftmaxPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

    vkCmdDispatch(cmdBuffer->get(), 1, numY, 1);

    return NO_ERROR;
}

ErrorCode VulkanSoftmax::onEncodeBuffer(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                        const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input = inputs[0];
    auto output = outputs[0];
    auto axis = mAxis;
    if (axis < 0) {
        axis += input->dimensions();
    }
    int inside = 1;
    int outside = 1;
    int mid = input->length(axis);
    for (int i = 0; i < axis; ++i) {
        outside *= input->length(i);
    }
    for (int i = axis + 1; i < input->dimensions(); ++i) {
        inside *= input->length(i);
    }
    {
        auto softmax = reinterpret_cast<BufferSoftmaxConst*>(mSoftmaxConstBuffer->map());
        ::memset(softmax, 0, sizeof(BufferSoftmaxConst));
        softmax->inside = inside;
        softmax->axis = mid;
        softmax->outside = outside;
        mSoftmaxConstBuffer->unmap();
    }

    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto inputBufferSize = input->elementSize() * sizeof(float);
    auto outputBufferSize = output->elementSize() * sizeof(float);
    mSource.buffer.reset(new VulkanBuffer(vkBn->getDynamicMemoryPool(), false, inputBufferSize, nullptr,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    mOutput.buffer.reset(new VulkanBuffer(vkBn->getDynamicMemoryPool(), false, outputBufferSize, nullptr,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));

    mSource.convert->encodeTensorToBuffer(input, mSource.buffer->buffer(), mSource.buffer->size(), 0,
                                          VulkanImageConverter::getTensorLinearFormat(input), cmdBuffer);
    cmdBuffer->barrierSource(mSource.buffer->buffer(), 0, mSource.buffer->size());

    mDescriptorSet.reset(mSoftmaxPipeline->createSet());
    mDescriptorSet->writeBuffer(mOutput.buffer->buffer(), 0, mOutput.buffer->size());
    mDescriptorSet->writeBuffer(mSource.buffer->buffer(), 1, mSource.buffer->size());
    mDescriptorSet->writeBuffer(mSoftmaxConstBuffer->buffer(), 2, mSoftmaxConstBuffer->size());
    mSoftmaxPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(outside, 8), UP_DIV(inside, 8), 1);

    cmdBuffer->barrierSource(mOutput.buffer->buffer(), 0, mOutput.buffer->size());
    mOutput.convert->encodeBufferToTensor(mOutput.buffer->buffer(), output, mOutput.buffer->size(), 0,
                                          VulkanImageConverter::getTensorLinearFormat(output), cmdBuffer);
    mSource.buffer->release();
    mOutput.buffer->release();
    return NO_ERROR;
}

class VulkanSoftmaxCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        auto input = inputs[0];

        uint32_t dimension = input->dimensions();
        if (dimension > 4) {
            return nullptr;
        }

        // Work out the reduce axis, taking various formats and dimensions into account.
        MNN_DATA_FORMAT format = VulkanImageConverter::getTensorLinearFormat(input);
        int axis = op->main_as_Axis()->axis();
        if (axis < 0) {
            axis = input->dimensions() + axis;
        }
        std::vector<uint32_t> axisMap;

        if (dimension == 4) {
            if (format == MNN_DATA_FORMAT_NCHW) {
                axisMap.assign({0, 3, 1, 2});
            } else {
                axisMap.assign({0, 1, 2, 3});
            }
        } else if (dimension == 3) {
            if (format == MNN_DATA_FORMAT_NCHW) {
                axisMap.assign({0, 3, 1});
            } else {
                axisMap.assign({0, 1, 3});
            }
        } else if (dimension == 2) {
            axisMap.assign({0, 3});
        } else if (dimension == 1) {
            axisMap.assign({3});
        } else {
            return nullptr;
        }
        uint32_t axisIndex = axisMap[axis];

        auto vkBn = static_cast<VulkanBackend*>(backend);
        auto imageLimit = vkBn->proty().limits.maxImageDimension2D;
        auto needsMultipleImages = [imageLimit](const Tensor* tensor) {
            auto shape = VulkanTensor::tensorShapeFormat(tensor);
            auto width = UP_DIV(shape[3], 4) * shape[2];
            auto height = shape[0] * shape[1];
            return width > imageLimit || height > imageLimit;
        };
        auto useBuffer = needsMultipleImages(input) || needsMultipleImages(outputs[0]);
        return new VulkanSoftmax(op, backend, axisIndex, useBuffer);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Softmax, new VulkanSoftmaxCreator);
    return true;
}();

} // namespace MNN
