//
//  VulkanPRelu.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanPRelu.hpp"
#include "VulkanUnary.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
namespace MNN {

struct GpuReluParam {
    ivec4 imgSize;
};

//--------------------------Prelu--------------------------//
VulkanPrelu::VulkanPrelu(Backend *bn, const Op *op) : VulkanBasicExecution(bn) {
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    };
    auto vulkanBn    = static_cast<VulkanBackend *>(bn);
    mPreluPipeline   = vulkanBn->getPipeline("glsl_preluWithChannel_comp",types);
    const auto prelu = op->main_as_PRelu();
    mGpuPreluParam = vulkanBn->allocUniform();
    int count = ALIGN_UP4(prelu->slope()->size());
    {
        std::shared_ptr<VulkanBuffer> slopeBuffer(new VulkanBuffer(
            vulkanBn->getMemoryPool(), false, sizeof(float) * count, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        auto slope = slopeBuffer->map();
        ::memset(slope, 0, count * sizeof(float));
        ::memcpy(slope, prelu->slope()->data(), prelu->slope()->size() * sizeof(float));
        slopeBuffer->unmap();
        mSlope = slopeBuffer;
    }
    mDescriptorSet.reset(mPreluPipeline->createSet());
}

VulkanPrelu::~VulkanPrelu() {
    auto extra = static_cast<VulkanBackend*>(backend());
    extra->recycleUniform(mGpuPreluParam);
}

ErrorCode VulkanPrelu::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const VulkanCommandPool::Buffer *cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];

    auto preluParam = reinterpret_cast<GpuReluParam *>(mGpuPreluParam->map());
    ::memset(preluParam, 0, sizeof(GpuReluParam));
    auto vkBn = static_cast<VulkanBackend *>(backend());

    const int channelDiv4  = UP_DIV(input->channel(), 4);
    auto planeSize = input->width() * input->height() * input->batch();
    preluParam->imgSize[0] = planeSize;
    preluParam->imgSize[1] = channelDiv4;
    preluParam->imgSize[2] = 1;
    preluParam->imgSize[3] = channelDiv4 * planeSize;
    mGpuPreluParam->unmap();
    auto total = planeSize * channelDiv4;

    auto vkOutput  = vkBn->getBuffer(output);
    auto vkInput   = vkBn->getBuffer(input);
    mDescriptorSet->writeBuffer(vkOutput, 0);
    mDescriptorSet->writeBuffer(vkInput, 1);
    mDescriptorSet->writeBuffer(mSlope->buffer(), 2, mSlope->size());

    mDescriptorSet->writeBuffer(mGpuPreluParam->buffer(), 3, mGpuPreluParam->size());
    mPreluPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(total, 256), 1, 1);
    return NO_ERROR;
}

class VulkanReluCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor*>& outputs, const MNN::Op *op, Backend *bn) const override {
        if (1 == op->main_as_PRelu()->slopeCount()) {
            return new VulkanUnary("RELU", bn, op->main_as_PRelu()->slope()->data()[0]);
        }
        return new VulkanPrelu(bn, op);
    }
};

static bool gr = []() {
    VulkanBackend::addCreator(OpType_PReLU, new VulkanReluCreator);
    return true;
}();

} // namespace MNN
