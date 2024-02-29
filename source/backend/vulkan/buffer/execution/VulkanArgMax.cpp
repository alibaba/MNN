//
//  VulkanArgMax.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanArgMax.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

struct ConstBuffer {
    ivec4 size;
};

VulkanArgMax::VulkanArgMax(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    mAxis                   = op->main_as_ArgMax()->axis();
    auto vkBn = (VulkanBackend*)backend();
    mConstBuffer = vkBn->allocUniform();
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    };
    if (op->type() == OpType_ArgMax) {
        mArgmaxPipeline =
            vkBn->getPipeline("glsl_argmax_comp", types);
    } else {
        MNN_ASSERT(op->type() == OpType_ArgMin);
        mArgmaxPipeline =
            vkBn->getPipeline("glsl_argmax_ARGMIN_comp", types);
    }
    mDescriptorSet.reset(mArgmaxPipeline->createSet());
}

VulkanArgMax::~VulkanArgMax() {
    auto vkBn = (VulkanBackend*)backend();
    vkBn->recycleUniform(mConstBuffer);
}

ErrorCode VulkanArgMax::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
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
    for (int i=axis+1; i<input->dimensions(); ++i) {
        inside *= input->length(i);
    }
    auto total = outside * inside;
    int outsideParallel = 1;
    int reduceAxis = 1;
    if (total >= 256) {
        reduceAxis = 1;
        outsideParallel = 256;
    } else if (total < 16) {
        reduceAxis = 256;
        outsideParallel = 1;
    } else {
        reduceAxis = 16;
        outsideParallel = 16;
    }
    
    // gpu param
    {
        auto Argmax = reinterpret_cast<ConstBuffer*>(mConstBuffer->map());
        Argmax->size[0] = inside;
        Argmax->size[1] = mid;
        Argmax->size[2] = outside;
        Argmax->size[3] = reduceAxis;
        mConstBuffer->unmap();
    }
    auto vkBn = static_cast<VulkanBackend*>(backend());
    mDescriptorSet->writeBuffer(vkBn->getBuffer(output), 0);
    mDescriptorSet->writeBuffer(vkBn->getBuffer(input), 1);
    mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 2, mConstBuffer->size());
    mArgmaxPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(total, outsideParallel), 1, 1);
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

} // namespace MNN
