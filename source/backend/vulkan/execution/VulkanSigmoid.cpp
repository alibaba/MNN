//
//  VulkanSigmoid.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanSigmoid.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {

struct VKSigmoidArgs {
    ivec2 size; /*[0]->width [1]->height*/
};

#define LOCAL_SIZE_X (16)
#define LOCAL_SIZE_Y (16)
#define LOCAL_SIZE_Z (1)

VulkanSigmoid::VulkanSigmoid(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto extra = static_cast<VulkanBackend*>(bn);

    mArgs = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false, sizeof(struct VKSigmoidArgs), nullptr,
                                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
}

VulkanSigmoid::~VulkanSigmoid() {
}

ErrorCode VulkanSigmoid::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const VulkanCommandPool::Buffer* cmdBuffer) {
    Tensor* input  = inputs[0];
    Tensor* output = outputs[0];

    const int channelDiv4 = UP_DIV(input->channel(), 4);

    struct VKSigmoidArgs* sigmoidArgs = reinterpret_cast<struct VKSigmoidArgs*>(mArgs->map());

    sigmoidArgs->size[0] = output->width();
    sigmoidArgs->size[1] = output->height();

    mArgs->flush(true, 0, sizeof(struct VKSigmoidArgs));
    mArgs->unmap();
    auto extra = static_cast<VulkanBackend*>(backend());

    std::vector<VkDescriptorType> image_types{
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    };
    mImagePipeline = extra->getPipeline("glsl_sigmoidimage_comp",
                                        /*glsl_sigmoidimage_comp, glsl_sigmoidimage_comp_len,*/ image_types);
    mDescriptorSet.reset(mImagePipeline->createSet());
    mDescriptorSet->writeBuffer(mArgs->buffer(), 0, mArgs->size());
    VulkanBackend* vkBn = (VulkanBackend*)backend();
    mDescriptorSet->writeImage((VkImageView)output->deviceId(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 1);
    mDescriptorSet->writeImage((VkImageView)input->deviceId(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
    mImagePipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    /*we do'nt use channel*/
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(output->width(), 16), UP_DIV(output->height(), 16),
                  UP_DIV(channelDiv4 * input->batch(), 1));
    return NO_ERROR;
}

class VulkanSigmoidCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override {
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
            return nullptr;
        }
        return new VulkanSigmoid(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Sigmoid, new VulkanSigmoidCreator);
    return true;
}();

} // namespace MNN
