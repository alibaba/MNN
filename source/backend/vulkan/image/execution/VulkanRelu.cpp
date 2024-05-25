//
//  VulkanRelu.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanRelu.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
namespace MNN {

struct GpuReluParam {
    ivec4 imgSize;
    vec4 slope;
};

//--------------------------relu--------------------------//
VulkanRelu::VulkanRelu(Backend *bn, const Op* op) : VulkanBasicExecution(bn) {
    auto vulkanBn = static_cast<VulkanBackend *>(bn);
    if (op->type() == OpType_ReLU6) {
        float minv = 0.0f;
        float maxv = 6.0f;
        if (nullptr != op->main_as_Relu6()) {
            minv = op->main_as_Relu6()->minValue();
            maxv = op->main_as_Relu6()->maxValue();
        }
        mSlope[0] = minv;
        mSlope[1] = maxv;
        mReluPipeline = vulkanBn->getPipeline("glsl_relu6_comp", {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER});
    } else {
        if (op->type() == OpType_ReLU) {
            mSlope[0] = op->main_as_Relu()->slope();
            mSlope[1] = op->main_as_Relu()->slope();
            mSlope[2] = op->main_as_Relu()->slope();
            mSlope[3] = op->main_as_Relu()->slope();
        } else {
            // PRELU
            auto slope = op->main_as_PRelu()->slope()->data()[0];
            mSlope[0] = slope;
            mSlope[1] = slope;
            mSlope[2] = slope;
            mSlope[3] = slope;
        }

        mReluPipeline = vulkanBn->getPipeline("glsl_relu_comp", {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER});
    }
}

VulkanRelu::~VulkanRelu() {
}

ErrorCode VulkanRelu::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                               const VulkanCommandPool::Buffer *cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];

    auto vkBn = (VulkanBackend *)backend();

    auto inputTensor = reinterpret_cast<VulkanTensor*>(input->deviceId());
    auto outputTensor = reinterpret_cast<VulkanTensor*>(output->deviceId());
    auto vkOutput = reinterpret_cast<VulkanTensor*>(output->deviceId());
    auto vkInput  = reinterpret_cast<VulkanTensor*>(input->deviceId());
    mDescriptorSet.resize(vkOutput->imageSize());
    mGpuReluParam.resize(vkOutput->imageSize());
    for (int i=0; i<vkOutput->imageSize(); ++i) {
        mGpuReluParam[i].reset(new VulkanBuffer(vkBn->getMemoryPool(), false, sizeof(GpuReluParam), nullptr,
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        auto reluParam = reinterpret_cast<GpuReluParam *>(mGpuReluParam[i]->map());
        ::memset(reluParam, 0, sizeof(GpuReluParam));
        reluParam->imgSize[0] = inputTensor->image(i)->width();
        reluParam->imgSize[1] = inputTensor->image(i)->height();
        reluParam->imgSize[2] = inputTensor->image(i)->depth();
        reluParam->imgSize[3] = 0;
        for (int v=0; v<4; ++v) {
            reluParam->slope[v]      = mSlope[v];
        }
        mGpuReluParam[i]->unmap();
        mDescriptorSet[i].reset(mReluPipeline->createSet());
        mDescriptorSet[i]->writeImage(outputTensor->image(i)->view(), vkBn->getCommonSampler()->get(),
                                VK_IMAGE_LAYOUT_GENERAL, 0);
        mDescriptorSet[i]->writeImage(inputTensor->image(i)->view(), vkBn->getCommonSampler()->get(),
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        mDescriptorSet[i]->writeBuffer(mGpuReluParam[i]->buffer(), 2, mGpuReluParam[i]->size());
        mReluPipeline->bind(cmdBuffer->get(), mDescriptorSet[i]->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(inputTensor->image(i)->width(), 16), UP_DIV(inputTensor->image(i)->height(), 16), 1);
    }
    return NO_ERROR;
}
//--------------------------Prelu--------------------------//
VulkanPrelu::VulkanPrelu(Backend *bn, const Op *op) : VulkanBasicExecution(bn) {
    std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    auto vulkanBn    = static_cast<VulkanBackend *>(bn);
    mPreluPipeline   = vulkanBn->getPipeline("glsl_preluWithChannel_comp",
                                           /*glsl_preluWithChannel_comp, glsl_preluWithChannel_comp_len,*/ types);
    const auto prelu = op->main_as_PRelu();
    mGpuPreluParam.reset(new VulkanBuffer(vulkanBn->getMemoryPool(), false, sizeof(GpuReluParam), nullptr,
                                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    int count = ALIGN_UP4(prelu->slope()->size());

    mSlope.reset(new VulkanImage(vulkanBn->getMemoryPool(), false, std::vector<int>{count / 4, 1}));
    {
        std::shared_ptr<VulkanBuffer> slopeBuffer(new VulkanBuffer(
            vulkanBn->getMemoryPool(), false, sizeof(float) * count, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        auto slope = slopeBuffer->map();
        ::memset(slope, 0, count * sizeof(float));
        ::memcpy(slope, prelu->slope()->data(), prelu->slope()->size() * sizeof(float));
        slopeBuffer->unmap();
        vulkanBn->copyBufferToImage(slopeBuffer.get(), mSlope.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
}

VulkanPrelu::~VulkanPrelu() {
}

ErrorCode VulkanPrelu::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const VulkanCommandPool::Buffer *cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];

    auto preluParam = reinterpret_cast<GpuReluParam *>(mGpuPreluParam->map());
    ::memset(preluParam, 0, sizeof(GpuReluParam));
    auto vkBn = static_cast<VulkanBackend *>(backend());

    const int channelDiv4  = UP_DIV(input->channel(), 4);
    preluParam->imgSize[0] = input->width();
    preluParam->imgSize[1] = input->height();
    preluParam->imgSize[2] = channelDiv4;
    preluParam->imgSize[3] = 0;
    mGpuPreluParam->flush(true, 0, sizeof(GpuReluParam));
    mGpuPreluParam->unmap();

    auto vkBackend = (VulkanBackend*)backend();
    auto vkOutput  = (VulkanTensor*)output->deviceId();
    auto vkInput   = (VulkanTensor*)input->deviceId();

    mDescriptorSet.reset(mPreluPipeline->createSet());
    mDescriptorSet->writeImage(((VulkanTensor*)output->deviceId())->image()->view(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 0);
    mDescriptorSet->writeImage(((VulkanTensor*)input->deviceId())->image()->view(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mDescriptorSet->writeImage((mSlope->view()), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
    mDescriptorSet->writeBuffer(mGpuPreluParam->buffer(), 3, mGpuPreluParam->size());

    vkOutput->image()->barrierWrite(cmdBuffer->get());
    vkInput->image()->barrierRead(cmdBuffer->get());
    mSlope->barrierRead(cmdBuffer->get());

    mPreluPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(input->width(), 16), UP_DIV(input->height(), 16), channelDiv4 * input->batch());
    return NO_ERROR;
}

class VulkanReluCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor*>& outputs, const MNN::Op *op, Backend *bn) const override {
        auto type  = op->type();
        if (OpType_ReLU6 == type) {
            return new VulkanRelu(bn, op);
        }
        if (OpType_ReLU == type) {
            return new VulkanRelu(bn, op);
        } else if (1 == op->main_as_PRelu()->slopeCount()) {
            return new VulkanRelu(bn, op);
        } else {
            return new VulkanPrelu(bn, op);
        }
        return nullptr;
    }
};

static bool gr = []() {
    VulkanBackend::addCreator(OpType_ReLU, new VulkanReluCreator);
    VulkanBackend::addCreator(OpType_PReLU, new VulkanReluCreator);
    VulkanBackend::addCreator(OpType_ReLU6, new VulkanReluCreator);
    return true;
}();

} // namespace MNN
