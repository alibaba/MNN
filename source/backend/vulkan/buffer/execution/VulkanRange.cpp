//
//  VulkanRange.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanRange.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

struct Param {
    ivec4 size;
};

VulkanRange::VulkanRange(halide_type_t type, Backend* backend) : VulkanBasicExecution(backend) {
    auto vkbackend = static_cast<VulkanBackend*>(backend);
    mParam = vkbackend->allocUniform();
    auto types = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    if (type.code == halide_type_int) {
        mPipeline = vkbackend->getPipeline("glsl_range_USE_INT_comp", types);
    } else {
        mPipeline = vkbackend->getPipeline("glsl_range_comp", types);
    }
    mDesSet.reset(mPipeline->createSet());
}

VulkanRange::~VulkanRange() {
    auto vkbackend = static_cast<VulkanBackend*>(backend());
    vkbackend->recycleUniform(mParam);
}

ErrorCode VulkanRange::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const VulkanCommandPool::Buffer* cmdBuffer) {
    // set param
    auto outSize = outputs[0]->elementSize();
    auto param = reinterpret_cast<Param*>(mParam->map());
    param->size[0] = outSize;
    param->size[1] = outSize;
    param->size[2] = outSize;
    param->size[3] = outSize;
    mParam->unmap();
    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto inputTensor0 = vkBn->getBuffer(inputs[0]);
    auto inputTensor1 = vkBn->getBuffer(inputs[2]);
    auto outputTensor = vkBn->getBuffer(outputs[0]);
    mDesSet->writeBuffer(outputTensor, 0);
    mDesSet->writeBuffer(inputTensor0, 1);
    mDesSet->writeBuffer(inputTensor1, 2);
    mDesSet->writeBuffer(mParam->buffer(), 3, mParam->size());

    mPipeline->bind(cmdBuffer->get(), mDesSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(outSize, 256), 1, 1);

    return NO_ERROR;
}

class VulkanRangeCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanRange(outputs[0]->getType(), bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Range, new VulkanRangeCreator);
    return true;
}();

} // namespace MNN
