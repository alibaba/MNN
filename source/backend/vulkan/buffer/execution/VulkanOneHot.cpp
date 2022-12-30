//
//  VulkanOneHot.cpp
//  MNN
//
//  Created by MNN on 2020/07/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanOneHot.hpp"
#include "core/Macro.h"
namespace MNN {
struct ConstBuffer {
    ivec4 inputSize;
};

VulkanOneHot::VulkanOneHot(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto extra = static_cast<VulkanBackend*>(bn);
    mAxis = op->main_as_OneHotParam()->axis();
    mConstBuffer = extra->allocUniform();
    mPipeline = extra->getPipeline("glsl_onehot_comp", {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    });
    mDescriptorSet.reset(mPipeline->createSet());
}
VulkanOneHot::~VulkanOneHot() {
    auto extra = static_cast<VulkanBackend*>(backend());
    extra->recycleUniform(mConstBuffer);
}

ErrorCode VulkanOneHot::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) {
    auto indices        = inputs[0];
    auto depthTensor    = inputs[1];
    auto onValueTensor  = inputs[2];
    auto offValueTensor = inputs[3];

    int axis = mAxis;
    if (axis < 0) {
        axis += outputs[0]->dimensions();
    }
    int outerSize = 1;
    for (int i = 0; i < axis; ++i) {
        outerSize *= indices->length(i);
    }
    const int depth       = outputs[0]->length(axis);
    const int innerSize   = indices->elementSize() / outerSize;
    const auto indicesPtr = indices->host<int>();
    auto total = outputs[0]->elementSize();

    auto dataType    = onValueTensor->getType();
    auto offDataType = offValueTensor->getType();
    auto vkBn = static_cast<VulkanBackend*>(backend());
    MNN_ASSERT(dataType == offDataType);
    // Set Const Buffer
    {
        auto pool = (ConstBuffer*)mConstBuffer->map();
        ::memset(pool, 0, sizeof(ConstBuffer));
        pool->inputSize[0]  = outerSize;
        pool->inputSize[1]  = depth;
        pool->inputSize[2]  = innerSize;
        pool->inputSize[3]  = total;
        mConstBuffer->unmap();
    }

    // Set Command Buffer
    {
        auto outputT = vkBn->getBuffer(outputs[0]);
        auto inputT = vkBn->getBuffer(indices);
        auto onT = vkBn->getBuffer(onValueTensor);
        auto offT = vkBn->getBuffer(offValueTensor);
        mDescriptorSet->writeBuffer(outputT, 0);
        mDescriptorSet->writeBuffer(inputT, 1);
        cmdBuffer->barrierSource(inputT);
        mDescriptorSet->writeBuffer(onT, 2);
        mDescriptorSet->writeBuffer(offT, 3);
        mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 4, mConstBuffer->size());
        mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(total, 256), 1, 1);
    }
    return NO_ERROR;
}

class VulkanOneHotCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new VulkanOneHot(op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_OneHot, new VulkanOneHotCreator);
    return true;
}();

} // namespace MNN
