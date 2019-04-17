//
//  VulkanElementWise.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanElementWise.hpp"
#include "Macro.h"

namespace MNN {

struct ConstBuffer {
    ivec4 imgSize;
    ivec4 stride;
};

VulkanElementWise::VulkanElementWise(EltwiseType type, Backend* bn) : Execution(bn) {
    auto extra = static_cast<VulkanBackend*>(bn);
    std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};

    switch (type) {
        case EltwiseType_SUM:
            mElemenWisePipeline = extra->getPipeline("glsl_elementwiseAdd_comp",
                                                     /*glsl_elementwiseAdd_comp, glsl_elementwiseAdd_comp_len,*/ types);
            break;

        case EltwiseType_PROD:
            mElemenWisePipeline = extra->getPipeline("glsl_elementwiseMul_comp",
                                                     /*glsl_elementwiseMul_comp, glsl_elementwiseMul_comp_len,*/ types);
            break;

        default:
            MNN_PRINT("Not Supported Eltwise Type: %d\n", type);
            MNN_ASSERT(false);
            break;
    }
    mConstBuffer.reset(new VulkanBuffer(extra->getMemoryPool(), false, sizeof(ConstBuffer), nullptr,
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
}

VulkanElementWise::~VulkanElementWise() {
}

ErrorCode VulkanElementWise::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto extra = static_cast<VulkanBackend*>(backend());
    for (auto& b : mBuffers) {
        extra->pushCommand(b->get());
    }

    return NO_ERROR;
}

ErrorCode VulkanElementWise::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(2 <= inputs.size());
    MNN_ASSERT(1 == outputs.size());

    auto input0 = inputs[0];
    auto input1 = inputs[1];
    MNN_ASSERT(input0->buffer().dim[1].flags == 1);
    MNN_ASSERT(input1->buffer().dim[1].flags == 1);
    auto output = outputs[0];

    const int iw     = input0->width();
    const int ih     = input0->height();
    const int icDiv4 = UP_DIV(input0->channel(), 4);
    const int ocDiv4 = UP_DIV(output->channel(), 4);

    auto elemenwiseSize = reinterpret_cast<ConstBuffer*>(mConstBuffer->map());
    ::memset(elemenwiseSize, 0, sizeof(ConstBuffer));

    elemenwiseSize->imgSize[0] = iw;
    elemenwiseSize->imgSize[1] = ih;
    elemenwiseSize->imgSize[2] = icDiv4;
    elemenwiseSize->imgSize[3] = input0->batch();

    mConstBuffer->flush(true, 0, sizeof(ConstBuffer));
    mConstBuffer->unmap();
    mSubDescriptorSets.clear();
    mBuffers.clear();

    auto vkbackend = static_cast<VulkanBackend*>(backend());
    auto sampler   = vkbackend->getCommonSampler()->get();
    std::shared_ptr<VulkanPipeline::DescriptorSet> descriptorSet(mElemenWisePipeline->createSet());
    mSubDescriptorSets.push_back(descriptorSet);
    const VulkanTensor* vkTensor = nullptr;
    vkTensor                     = vkbackend->findTensor(output->deviceId());

    descriptorSet->writeImage(vkTensor->image()->view(), sampler, VK_IMAGE_LAYOUT_GENERAL, 0);
    descriptorSet->writeImage((VkImageView)input0->deviceId(), sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    descriptorSet->writeImage((VkImageView)input1->deviceId(), sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
    descriptorSet->writeBuffer(mConstBuffer->buffer(), 3, mConstBuffer->size());
    {
        std::shared_ptr<VulkanCommandPool::Buffer> cmdBuffer(
            const_cast<VulkanCommandPool::Buffer*>(vkbackend->getPool().allocBuffer()));
        cmdBuffer->begin(0);
        mElemenWisePipeline->bind(cmdBuffer->get(), descriptorSet->get());
        auto input0Vk = vkbackend->findTensor(input0->deviceId());
        auto input1Vk = vkbackend->findTensor(input1->deviceId());
        cmdBuffer->barrierImage(input0Vk->image()->get(), VK_IMAGE_LAYOUT_GENERAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        cmdBuffer->barrierImage(input1Vk->image()->get(), VK_IMAGE_LAYOUT_GENERAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(iw, 8), UP_DIV(ih, 8), UP_DIV(ocDiv4 * output->batch(), 4));
        cmdBuffer->end();
        mBuffers.push_back(cmdBuffer);
    }

    for (int i = 2; i < inputs.size(); ++i) {
        std::shared_ptr<VulkanCommandPool::Buffer> cmdBuffer(
            const_cast<VulkanCommandPool::Buffer*>(vkbackend->getPool().allocBuffer()));
        cmdBuffer->begin(0);
        auto inputI = inputs[i];
        std::shared_ptr<VulkanPipeline::DescriptorSet> subDescriptorSet(mElemenWisePipeline->createSet());
        subDescriptorSet->writeImage(vkTensor->image()->view(), sampler, VK_IMAGE_LAYOUT_GENERAL, 0);
        subDescriptorSet->writeImage(vkTensor->image()->view(), sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        subDescriptorSet->writeImage((VkImageView)inputI->deviceId(), sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     2);
        subDescriptorSet->writeBuffer(mConstBuffer->buffer(), 3, mConstBuffer->size());
        mElemenWisePipeline->bind(cmdBuffer->get(), subDescriptorSet->get());
        auto input0Vk = vkTensor;
        auto input1Vk = vkbackend->findTensor(inputI->deviceId());
        cmdBuffer->barrierImage(input0Vk->image()->get(), VK_IMAGE_LAYOUT_GENERAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        cmdBuffer->barrierImage(input1Vk->image()->get(), VK_IMAGE_LAYOUT_GENERAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vkCmdDispatch(cmdBuffer->get(), UP_DIV(iw, 8), UP_DIV(ih, 8), UP_DIV(ocDiv4 * output->batch(), 4));
        mSubDescriptorSets.push_back(subDescriptorSet);
        cmdBuffer->end();
        mBuffers.push_back(cmdBuffer);
    }

    return NO_ERROR;
}

class VulkanElementWiseCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op,
                                Backend* backend) const override {
        const auto elementWiseType = op->main_as_Eltwise();
        return new VulkanElementWise(elementWiseType->type(), backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Eltwise, new VulkanElementWiseCreator);
    return true;
}();

} // namespace MNN
