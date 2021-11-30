//
//  VulkanGridSample.cpp
//  MNN
//
//  Created by MNN on 2021/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanGridSample.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

struct GpuGridSampleParam {
    ivec4 outImgSize;
    ivec2 inShape;
    ivec2 outShape;
    bool alignCorners;
};

VulkanGridSample::VulkanGridSample(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto vulkanBn = static_cast<VulkanBackend *>(bn);
    mGridSampleParam.reset(new VulkanBuffer(vulkanBn->getMemoryPool(), false, sizeof(GpuGridSampleParam), nullptr,
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));

    mAlignCorners = op->main_as_GridSample()->alignCorners();

    std::string prefix;
    if (0 == op->main_as_GridSample()->mode()) { // SampleMode_BILINEAR
        prefix = "glsl_gridSampleBilinear_";
    } else {
        prefix = "glsl_gridSampleNearest_";
    }

    std::string padding_mode = "";
    if (0 == op->main_as_GridSample()->paddingMode()) { // BorderMode_ZEROS
        padding_mode = "PAD_MODE_ZEROS_";
    }

    std::string posfix = "comp";

    auto types = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    };
    
    mGridSamplePipeline = vulkanBn->getPipeline(prefix + padding_mode + posfix, types);
}

VulkanGridSample::~VulkanGridSample() {
}

ErrorCode VulkanGridSample::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto grid   = inputs[1];
    auto output = outputs[0];
    
    auto vkBn = (VulkanBackend *)backend();
    
    auto inputTensor     = reinterpret_cast<VulkanTensor*>(input->deviceId());
    auto gridTensor      = reinterpret_cast<VulkanTensor*>(grid->deviceId());
    auto outputTensor    = reinterpret_cast<VulkanTensor*>(output->deviceId());
    auto gridSampleParam = reinterpret_cast<GpuGridSampleParam*>(mGridSampleParam->map());
    
    outputTensor->image()->barrierWrite(cmdBuffer->get());
    inputTensor->image()->barrierRead(cmdBuffer->get());
    gridTensor->image()->barrierRead(cmdBuffer->get());

    ::memset(gridSampleParam, 0, sizeof(GpuGridSampleParam));

    gridSampleParam->outImgSize[0] = outputTensor->image()->width();
    gridSampleParam->outImgSize[1] = outputTensor->image()->height();
    gridSampleParam->outImgSize[2] = outputTensor->image()->depth();
    gridSampleParam->outImgSize[3] = 0;

    gridSampleParam->inShape[0]  = input->width();
    gridSampleParam->inShape[1]  = input->height();
    gridSampleParam->outShape[0] = output->width();
    gridSampleParam->outShape[1] = output->height();

    gridSampleParam->alignCorners = mAlignCorners;
    
    mGridSampleParam->unmap();

    mDescriptorSet.reset(mGridSamplePipeline->createSet());
    mDescriptorSet->writeImage(outputTensor->image()->view(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 0);
    mDescriptorSet->writeImage(inputTensor->image()->view(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mDescriptorSet->writeImage(gridTensor->image()->view(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
    mDescriptorSet->writeBuffer(mGridSampleParam->buffer(), 3, mGridSampleParam->size());
    mGridSamplePipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(outputTensor->image()->width(), 16), UP_DIV(outputTensor->image()->height(), 16), 1);

    return NO_ERROR;
}

class VulkanGridSampleCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanGridSample(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_GridSample, new VulkanGridSampleCreator);
    return true;
}();

} // namespace MNN
