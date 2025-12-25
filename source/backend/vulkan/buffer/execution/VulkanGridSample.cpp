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
    ivec4 inShape;
    ivec4 outShape;
    bool alignCorners;
};

VulkanGridSample::VulkanGridSample(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto vulkanBn = static_cast<VulkanBackend *>(bn);
    mGridSampleParam.reset(new VulkanBuffer(vulkanBn->getMemoryPool(), false, sizeof(GpuGridSampleParam), nullptr,
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));

    mAlignCorners = op->main_as_GridSample()->alignCorners();

    std::string pKey = "glsl_";
    if (0 == op->main_as_GridSample()->mode()) { // SampleMode_BILINEAR
        pKey += "gridSampleBilinear_";
    } else {
        pKey += "gridSampleNearest_";
    }

    if (0 == op->main_as_GridSample()->paddingMode()) { // BorderMode_ZEROS
        pKey += "PAD_MODE_ZEROS_";
    }

    if (vulkanBn->useFP16()) {
        pKey += "FP16_";
    }

    pKey += "comp";

    auto types = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    };
    
    mGridSamplePipeline = vulkanBn->getPipeline(pKey, types);
    mDescriptorSet.reset(mGridSamplePipeline->createSet());
}

VulkanGridSample::~VulkanGridSample() {
}

ErrorCode VulkanGridSample::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto grid   = inputs[1];
    auto output = outputs[0];
    
    auto vkBn = (VulkanBackend *)backend();
    
    auto inputTensor     = vkBn->getBuffer(input);
    auto gridTensor      = vkBn->getBuffer(grid);
    auto outputTensor    = vkBn->getBuffer(output);
    auto gridSampleParam = reinterpret_cast<GpuGridSampleParam*>(mGridSampleParam->map());
    
    ::memset(gridSampleParam, 0, sizeof(GpuGridSampleParam));
    auto iC4 = UP_DIV(input->channel(), 4);
    auto oC4 = UP_DIV(output->channel(), 4);

    gridSampleParam->inShape[0]  = input->width();
    gridSampleParam->inShape[1]  = input->height();
    gridSampleParam->inShape[2]  = iC4;
    gridSampleParam->inShape[3]  = input->batch();
    gridSampleParam->outShape[0] = output->width();
    gridSampleParam->outShape[1] = output->height();
    gridSampleParam->outShape[2]  = oC4;
    gridSampleParam->outShape[3]  = output->batch();
    int total = 1;
    for (int i=0; i<4; ++i) {
        total *= gridSampleParam->outShape[i];
    }

    gridSampleParam->alignCorners = mAlignCorners;
    
    mGridSampleParam->unmap();
    int width = output->width();
    int height = output->height();

    mDescriptorSet->writeBuffer(outputTensor, 0);
    mDescriptorSet->writeBuffer(inputTensor, 1);
    mDescriptorSet->writeBuffer(gridTensor, 2);
    mDescriptorSet->writeBuffer(mGridSampleParam->buffer(), 3, mGridSampleParam->size());
    mGridSamplePipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(total, 256), 1, 1);

    return NO_ERROR;
}

class VulkanGridSampleCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        MNN_ASSERT(outputs[0]->getType().code == halide_type_float);
        return new VulkanGridSample(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_GridSample, new VulkanGridSampleCreator);
    return true;
}();

} // namespace MNN
