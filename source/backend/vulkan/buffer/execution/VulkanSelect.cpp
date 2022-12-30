//
//  VulkanSelect.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanSelect.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

struct Param {
    ivec4 size;
};

VulkanSelect::VulkanSelect(const Op* op, Backend* backend) : VulkanBasicExecution(backend) {
    auto vkbackend = static_cast<VulkanBackend*>(backend);
    mParam         = std::make_shared<VulkanBuffer>(vkbackend->getMemoryPool(), false, sizeof(Param), nullptr,
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    auto types = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    mPipeline = vkbackend->getPipeline("glsl_select_comp", types);
    mDesSet.reset(mPipeline->createSet());
}

VulkanSelect::~VulkanSelect() {
}


ErrorCode VulkanSelect::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const VulkanCommandPool::Buffer* cmdBuffer) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    // set param
    auto inSize1 = inputs[1]->elementSize();
    auto inSize2 = inputs[2]->elementSize();
    auto outSize = outputs[0]->elementSize();
    MNN_ASSERT(inputs[0]->elementSize() == outSize);
    MNN_ASSERT(inSize1 == 1 || inSize1 == outSize);
    MNN_ASSERT(inSize2 == 1 || inSize2 == outSize);

    auto param = reinterpret_cast<Param*>(mParam->map());
    param->size[0] = outSize;
    param->size[1] = inSize1;
    param->size[2] = inSize2;
    param->size[3] = outSize;
    mParam->unmap();
    auto inputTensor0 = vkBn->getBuffer(inputs[0]);
    auto inputTensor1 = vkBn->getBuffer(inputs[1]);
    auto inputTensor2 = vkBn->getBuffer(inputs[2]);
    auto outputTensor = vkBn->getBuffer(outputs[0]);
    mDesSet->writeBuffer(outputTensor, 0);
    mDesSet->writeBuffer(inputTensor0, 1);
    mDesSet->writeBuffer(inputTensor1, 2);
    mDesSet->writeBuffer(inputTensor2, 3);
    mDesSet->writeBuffer(mParam->buffer(), 4, mParam->size());

    mPipeline->bind(cmdBuffer->get(), mDesSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(outSize, 256), 1, 1);

    return NO_ERROR;
}

class VulkanSelectCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanSelect(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Select, new VulkanSelectCreator);
    return true;
}();

} // namespace MNN
