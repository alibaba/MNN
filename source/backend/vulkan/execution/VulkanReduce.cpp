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
VulkanReduce::VulkanReduce(const std::string& name, const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    mOp = op;
    auto vkBn = (VulkanBackend*)backend();
    mPipeline = vkBn->getPipeline(name, {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    });
}
VulkanReduce::~VulkanReduce() {
    
}
struct constBuffer {
    int w;//inside
    int h;//axis
    int c;//outside
    float k;//For mean
};
ErrorCode VulkanReduce::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    auto vkBn = (VulkanBackend*)backend();
    auto reduceDims = OpCommonUtils::computeReduceDims(inputs, mOp);
    MNN_ASSERT(!reduceDims.empty());
    // TODO: Divde large axis
    
    mMidBuffers.resize(reduceDims.size() - 1);
    mUnits.resize(reduceDims.size());
    std::shared_ptr<VulkanBuffer> preBuffer;
    // Alloc
    for (int i=0; i<reduceDims.size(); ++i) {
        mUnits[i].mConstBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), true, sizeof(constBuffer), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        auto& reduceDim = reduceDims[i];
        auto inside = std::get<2>(reduceDim);
        auto outside = std::get<0>(reduceDim);
        auto axis = std::get<1>(reduceDim);
        auto buffer = (constBuffer*)mUnits[i].mConstBuffer->map();
        buffer->w = inside;
        buffer->h = axis;
        buffer->c = outside;
        buffer->k = 1.0f/(float)axis;
        mUnits[i].mConstBuffer->unmap();
        std::shared_ptr<VulkanBuffer> newBuffer;
        if (i < reduceDims.size() - 1) {
            newBuffer.reset(new VulkanBuffer(vkBn->getDynamicMemoryPool(), false, outside * inside * sizeof(float), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
            mMidBuffers[i] = newBuffer;
        }
        if (preBuffer != nullptr) {
            preBuffer->release();
        }
        preBuffer = newBuffer;
    }
    
    // Encode
    auto srcBuffer = (VkBuffer)inputs[0]->deviceId();
    auto srcSize = inputs[0]->size();
    for (int i=0; i<reduceDims.size()-1; ++i) {
        auto& reduceDim = reduceDims[i];
        auto inside = std::get<2>(reduceDim);
        auto outside = std::get<0>(reduceDim);
        auto total = inside * outside;
        auto& u = mUnits[i];
        u.mDescriptorSet.reset(mPipeline->createSet());
        u.mDescriptorSet->writeBuffer(srcBuffer, 0, srcSize);
        u.mDescriptorSet->writeBuffer(mMidBuffers[i]->buffer(), 1, mMidBuffers[i]->size());
        u.mDescriptorSet->writeBuffer(u.mConstBuffer->buffer(), 2, u.mConstBuffer->size());
        
        mPipeline->bind(cmdBuffer->get(), u.mDescriptorSet->get());
        cmdBuffer->barrierSource(srcBuffer, 0, srcSize);
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(total, 256), 1, 1);
        srcBuffer = mMidBuffers[i]->buffer();
        srcSize = mMidBuffers[i]->size();
    }
    // Encode Last
    {
        auto& reduceDim = reduceDims[reduceDims.size()-1];
        auto inside = std::get<2>(reduceDim);
        auto outside = std::get<0>(reduceDim);
        auto total = inside * outside;
        auto& u = mUnits[reduceDims.size()-1];
        u.mDescriptorSet.reset(mPipeline->createSet());
        u.mDescriptorSet->writeBuffer(srcBuffer, 0, srcSize);
        u.mDescriptorSet->writeBuffer((VkBuffer)outputs[0]->deviceId(), 1, outputs[0]->size());
        u.mDescriptorSet->writeBuffer(u.mConstBuffer->buffer(), 2, u.mConstBuffer->size());
        mPipeline->bind(cmdBuffer->get(), u.mDescriptorSet->get());
        cmdBuffer->barrierSource(srcBuffer, 0, srcSize);
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(total, 256), 1, 1);
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
