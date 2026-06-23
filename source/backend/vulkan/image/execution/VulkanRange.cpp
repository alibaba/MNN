//
//  VulkanRange.cpp
//  MNN
//
//  Created by MNN on 2026/06/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanRange.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "VulkanBackend.hpp"

namespace MNN {

struct GpuRangeParam {
    ivec4 size; 
};

VulkanRange::VulkanRange(Backend* bn) : VulkanBasicExecution(bn) {
    auto vkBn = static_cast<VulkanBackend*>(bn);
    mPipeline = vkBn->getPipeline("glsl_range_comp", {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER});
}

VulkanRange::~VulkanRange() {
}

ErrorCode VulkanRange::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const VulkanCommandPool::Buffer* cmdBuffer) {
    auto outVT = reinterpret_cast<VulkanTensor*>(outputs[0]->deviceId());
    auto startVT = reinterpret_cast<VulkanTensor*>(inputs[0]->deviceId());
    auto deltaVT = reinterpret_cast<VulkanTensor*>(inputs[2]->deviceId());
    auto imageSize = outVT->imageSize();
    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto sampler = vkBn->getCommonSampler()->get();

    size_t oldPara = mParams.size();
    if (oldPara != imageSize) {
        mParams.resize(imageSize);
        mDescriptorSets.resize(imageSize);
        for (size_t i = oldPara; i < imageSize; ++i) {
            mParams[i] = std::make_shared<VulkanBuffer>(
                vkBn->getMemoryPool(), false, sizeof(GpuRangeParam),
                nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            mDescriptorSets[i].reset(mPipeline->createSet());
        }
    }

    auto startImg = startVT->image();
    auto deltaImg = deltaVT->image();
    startImg->barrierRead(cmdBuffer->get());
    deltaImg->barrierRead(cmdBuffer->get());
    int pixelOffset = 0;
    for(int i = 0; i < imageSize; ++i){
        auto outImg = outVT->image(i);
        auto total  = outImg->width() * outImg->height();

        auto p = reinterpret_cast<GpuRangeParam*>(mParams[i]->map());
        ::memset(p, 0, sizeof(GpuRangeParam));
        p->size[0] = pixelOffset;  
        p->size[3] = total; 
        mParams[i]->flush(true, 0, sizeof(GpuRangeParam));
        mParams[i]->unmap();

        auto desSet = mDescriptorSets[i];
        desSet->writeImage(outImg->view(),   sampler, VK_IMAGE_LAYOUT_GENERAL,                  0);
        desSet->writeImage(startImg->view(), sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        desSet->writeImage(deltaImg->view(), sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
        desSet->writeBuffer(mParams[i]->buffer(), 3, mParams[i]->size());

        outImg->barrierWrite(cmdBuffer->get());

        mPipeline->bind(cmdBuffer->get(), desSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(total, 256), 1, 1);

        pixelOffset += total;
    }

    return ErrorCode::NO_ERROR;
}

class VulkanRangeCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                           const MNN::Op* op, Backend* bn) const override {
        if (outputs[0]->getType().code != halide_type_float) {
            return nullptr;
        }
        return new VulkanRange(bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Range, new VulkanRangeCreator);
    return true;
}();

}// namespace MNN
