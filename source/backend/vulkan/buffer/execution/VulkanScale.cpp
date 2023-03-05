//
//  VulkanScale.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanScale.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

struct gpuScaleParam {
    ivec4 imgSize;
};

VulkanScale::VulkanScale(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    const auto scale   = op->main_as_Scale();
    const int channels = scale->scaleData()->size();

    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    };

    auto extra = static_cast<VulkanBackend*>(bn);

    mScalePipeline = extra->getPipeline("glsl_scale_comp", types);
    mScaleParam    = extra->allocUniform();
    auto channelsAlign = ALIGN_UP4(channels);
    mScaleBuffer   = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false, sizeof(float) * channelsAlign,
                                                  nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    mBiasBuffer   = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false, sizeof(float) * channelsAlign,
                                                  nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    {
        auto ptr = (float*)mScaleBuffer->map();
        ::memcpy(ptr, scale->scaleData()->data(), channels* sizeof(float));
        mScaleBuffer->unmap();
    }
    {
        auto ptr = (float*)mBiasBuffer->map();
        ::memcpy(ptr, scale->biasData()->data(), channels* sizeof(float));
        mBiasBuffer->unmap();
    }
    mDescriptorSet.reset(mScalePipeline->createSet());
}

VulkanScale::~VulkanScale() {
    auto extra = static_cast<VulkanBackend*>(backend());
    extra->recycleUniform(mScaleParam);
}

ErrorCode VulkanScale::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];

    MNN_ASSERT(MNN_DATA_FORMAT_NC4HW4 == TensorUtils::getDescribe(input)->dimensionFormat);

    auto scaleP = reinterpret_cast<gpuScaleParam*>(mScaleParam->map());
    ::memset(scaleP, 0, sizeof(gpuScaleParam));

    const int channelDiv4 = UP_DIV(input->channel(), 4);
    auto planeSize = input->width() * input->height() * input->batch();
    auto totalSize = planeSize * channelDiv4;

    scaleP->imgSize[0] = planeSize;
    scaleP->imgSize[1] = channelDiv4;
    scaleP->imgSize[2] = channelDiv4;
    scaleP->imgSize[3] = totalSize;
    mScaleParam->unmap();
    auto extra = static_cast<VulkanBackend*>(backend());

    mDescriptorSet->writeBuffer(extra->getBuffer(output), 0);
    mDescriptorSet->writeBuffer(extra->getBuffer(input), 1);
    mDescriptorSet->writeBuffer(mScaleBuffer->buffer(), 2, mScaleBuffer->size());
    mDescriptorSet->writeBuffer(mBiasBuffer->buffer(), 3, mBiasBuffer->size());
    mDescriptorSet->writeBuffer(mScaleParam->buffer(), 4, mScaleParam->size());
    mScalePipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalSize, 256), 1, 1);

    return NO_ERROR;
}

class VulkanScaleCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanScale(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Scale, new VulkanScaleCreator);
    return true;
}();

} // namespace MNN
