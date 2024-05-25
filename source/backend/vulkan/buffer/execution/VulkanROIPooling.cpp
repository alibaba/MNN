//
//  VulkanROIPooling.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanROIPooling.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
namespace MNN {
struct GpuParam {
    ivec4 inputImgSize;
    ivec4 outputImgSize;
    float spatialScale;
};

VulkanROIPooling::VulkanROIPooling(Backend* bn, const float SpatialScale)
    : VulkanBasicExecution(bn), mSpatialScale(SpatialScale) {
    std::vector<VkDescriptorType> VulkanROIPoolingTypes{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    auto extra                = static_cast<VulkanBackend*>(bn);
    mVulkanROIPoolingPipeline = extra->getPipeline(
        "glsl_roipooling_comp", VulkanROIPoolingTypes);
    mParamBuffer = extra->allocUniform();
    mDescriptorSet.reset(mVulkanROIPoolingPipeline->createSet());
}
VulkanROIPooling::~VulkanROIPooling() {
    auto extra = static_cast<VulkanBackend*>(backend());
    extra->recycleUniform(mParamBuffer);
}
ErrorCode VulkanROIPooling::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                     const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input            = inputs[0];
    auto roi              = inputs[1];
    auto output           = outputs[0];
    const int channelDiv4 = UP_DIV(input->channel(), 4);
    auto extra = static_cast<VulkanBackend*>(backend());

    auto VulkanROIPoolingParam = reinterpret_cast<GpuParam*>(mParamBuffer->map());
    ::memset(VulkanROIPoolingParam, 0, sizeof(GpuParam));
    VulkanROIPoolingParam->inputImgSize[0]  = input->width();
    VulkanROIPoolingParam->inputImgSize[1]  = input->height();
    VulkanROIPoolingParam->inputImgSize[2]  = channelDiv4;
    VulkanROIPoolingParam->inputImgSize[3]  = input->batch();
    VulkanROIPoolingParam->outputImgSize[0] = output->width();
    VulkanROIPoolingParam->outputImgSize[1] = output->height();
    VulkanROIPoolingParam->outputImgSize[2] = channelDiv4;
    VulkanROIPoolingParam->outputImgSize[3] = output->batch();
    VulkanROIPoolingParam->spatialScale     = mSpatialScale;
    mParamBuffer->unmap();

    mDescriptorSet->writeBuffer(extra->getBuffer(output), 0);
    mDescriptorSet->writeBuffer(extra->getBuffer(input), 1);
    mDescriptorSet->writeBuffer(extra->getBuffer(roi), 2);
    mDescriptorSet->writeBuffer(mParamBuffer->buffer(), 3, mParamBuffer->size());

    mVulkanROIPoolingPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(output->width(), 8), UP_DIV(output->height(), 8),
                  channelDiv4 * output->batch());

    return NO_ERROR;
}

class VulkanROIPoolingCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        if (TensorUtils::getDescribe(inputs[1])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            // Don't support old op version
            return nullptr;
        }
        return new VulkanROIPooling(bn, op->main_as_RoiParameters()->spatialScale());
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_ROIPooling, new VulkanROIPoolingCreator);
    return true;
}();

} // namespace MNN
