//
//  VulkanReduce.cpp
//  MNN
//
//  Created by MNN on 2020/03/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanReduce.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/Macro.h"
namespace MNN {
struct constBuffer {
    int w;//inside
    int h;//axis
    int c;//outside
    float k;//For mean
};
VulkanReduce::VulkanReduce(const std::string& name, const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto vkBn = (VulkanBackend*)backend();
    mOp = op;
    mPipeline = vkBn->getPipeline(name, {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    });
    mConstBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, sizeof(constBuffer), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    mDescriptorSet.reset(mPipeline->createSet());
}

VulkanReduce::~VulkanReduce() {
    // Do nothing
}

ErrorCode VulkanReduce::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    auto ptr = reinterpret_cast<constBuffer*>(mConstBuffer->map());
    ::memset(ptr, 0, sizeof(constBuffer));
    auto axisPos = mOp->main_as_ReductionParam()->dim()->data()[0];
    auto axis = inputs[0]->length(axisPos);
    int inside = 1;
    for (int i=axisPos+1; i<inputs[0]->dimensions(); ++i) {
        inside *= inputs[0]->length(i);
    }
    int outside = 1;
    for (int i=0; i<axisPos; ++i) {
        outside *= inputs[0]->length(i);
    }
    ptr->c = outside;
    ptr->h = axis;
    ptr->w = inside;
    ptr->k = 1.0f/(float)axis;
    auto total = outside * inside;
    auto output = outputs[0];
    auto input = inputs[0];
    mConstBuffer->unmap();
    auto vkBn = static_cast<VulkanBackend*>(backend());
    {
        int bufferSize = sizeof(float);
        for (int i=0; i<input->dimensions(); ++i) {
            bufferSize *= input->length(i);
        }
        mSource.buffer.reset(new VulkanBuffer(vkBn->getDynamicMemoryPool(),
                                           false, bufferSize, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        mSource.convert.reset(new VulkanImageConverter(vkBn));
    }
    {
        int bufferSize = sizeof(float);
        for (int i=0; i<output->dimensions(); ++i) {
            bufferSize *= output->length(i);
        }
        mOutput.convert.reset(new VulkanImageConverter(vkBn));
        mOutput.buffer.reset(new VulkanBuffer(vkBn->getDynamicMemoryPool(), false, bufferSize, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    }
    
    // Encode
    mSource.convert->encodeTensorToBuffer(input, mSource.buffer->buffer(), mSource.buffer->size(), 0, VulkanImageConverter::getTensorLinearFormat(input), cmdBuffer);

    mDescriptorSet->writeBuffer(mOutput.buffer->buffer(), 0, mOutput.buffer->size());
    mDescriptorSet->writeBuffer(mSource.buffer->buffer(), 1, mSource.buffer->size());
    mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 2, mConstBuffer->size());
    cmdBuffer->barrierSource(mSource.buffer->buffer(), 0, mSource.buffer->size());
    mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(total, 256), 1, 1);
    cmdBuffer->barrierSource(mOutput.buffer->buffer(), 0, mOutput.buffer->size());
    mOutput.convert->encodeBufferToTensor(mOutput.buffer->buffer(), output, mOutput.buffer->size(), 0, VulkanImageConverter::getTensorLinearFormat(output), cmdBuffer);
    {
        mSource.buffer->release();
        mOutput.buffer->release();
    }
    return NO_ERROR;
}

static std::string _getShaderName(const Op* op) {
    std::string prefix = "glsl_reduce_";
    std::string posfix = "_comp";
    std::string mid = "";
    switch (op->main_as_ReductionParam()->operation()) {
        case ReductionType_SUM:
            mid = "SUM";
            break;
        case ReductionType_MEAN:
            mid = "MEAN";
            break;
        case ReductionType_MAXIMUM:
            mid = "VMAX";
            break;
        case ReductionType_MINIMUM:
            mid = "VMIN";
            break;
        case ReductionType_PROD:
            mid = "PROD";
            break;
        default:
            break;
    }
    if (mid.empty()) {
        return mid;
    }
    return prefix + mid + posfix;
}
class VulkanReduceCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        auto input0 = inputs[0];
        if (input0->getType().code != halide_type_float) {
            return nullptr;
        }
        auto shader = _getShaderName(op);
        if (shader.empty()) {
            return nullptr;
        }
        return new VulkanReduce(shader, op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Reduction, new VulkanReduceCreator);
    return true;
}();
}
