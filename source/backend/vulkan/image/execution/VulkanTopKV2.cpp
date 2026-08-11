//
//  VulkanTopKV2.cpp
//  MNN
//
//  Vulkan image-mode implementation of TopKV2, directly operating on images.
//

#include "VulkanTopKV2.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

struct GpuTopKV2Param {
    int rowSize;
    int k;
    int numRows;
    int C4; // UP_DIV(rowSize, 4)
};

VulkanTopKV2::VulkanTopKV2(const Op* op, Backend* bn, int k) : VulkanBasicExecution(bn) {
    auto vkBn = (VulkanBackend *)backend();

    mK = k;
    mLargest = true;
    auto param = op->main_as_TopKV2();
    if (nullptr != param) {
        mLargest = param->largest();
    }

    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,        // output values (float image)
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,        // output indices (int image)
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, // input (float sampler)
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,       // params
    };

    if (mLargest) {
        mPipeline = vkBn->getPipeline("glsl_topkv2_SORT_DESC_comp", types);
    } else {
        mPipeline = vkBn->getPipeline("glsl_topkv2_comp", types);
    }

    mGpuParam.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, sizeof(GpuTopKV2Param), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    mDescriptorSet.reset(mPipeline->createSet());
}

VulkanTopKV2::~VulkanTopKV2() {
}

ErrorCode VulkanTopKV2::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const VulkanCommandPool::Buffer* cmdBuffer) {
    auto vkBn = (VulkanBackend*)backend();
    auto input = inputs[0];
    auto outputValue = outputs[0];
    auto outputIndex = outputs[1];

    const int rowSize = input->length(input->dimensions() - 1);
    if (rowSize <= 0) {
        MNN_PRINT("VulkanTopKV2: rowSize <= 0 (%d), skip execution with output uninitialized\n", rowSize);
        return NO_ERROR;
    }
    const int numRows = input->elementSize() / rowSize;
    const int k = mK;

    // Set GPU params
    auto topkParam = reinterpret_cast<GpuTopKV2Param*>(mGpuParam->map());
    topkParam->rowSize = rowSize;
    topkParam->k = k;
    topkParam->numRows = numRows;
    topkParam->C4 = UP_DIV(rowSize, 4);
    mGpuParam->unmap();

    // Get VulkanTensor image handles
    auto vkInput = reinterpret_cast<VulkanTensor*>(input->deviceId());
    auto vkOutValue = reinterpret_cast<VulkanTensor*>(outputValue->deviceId());
    auto vkOutIndex = reinterpret_cast<VulkanTensor*>(outputIndex->deviceId());

    // Write descriptor set with images directly
    mDescriptorSet.reset(mPipeline->createSet());
    mDescriptorSet->writeImage(vkOutValue->image()->view(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 0);
    mDescriptorSet->writeImage(vkOutIndex->image()->view(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 1);
    mDescriptorSet->writeImage(vkInput->image()->view(), vkBn->getCommonSampler()->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
    mDescriptorSet->writeBuffer(mGpuParam->buffer(), 3, mGpuParam->size());

    // Barriers
    vkOutValue->image()->barrierWrite(cmdBuffer->get());
    vkOutIndex->image()->barrierWrite(cmdBuffer->get());
    vkInput->image()->barrierRead(cmdBuffer->get());

    // Dispatch: x=1 (workgroup handles 128 threads internally), y=numRows, z=1
    mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), 1, numRows, 1);

    return NO_ERROR;
}

class VulkanTopKV2Creator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        if (inputs.size() < 2 || outputs.size() != 2) {
            return nullptr;
        }
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            return nullptr;
        }
        const int k = inputs[1]->host<int32_t>()[0];
        return new VulkanTopKV2(op, backend, k);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_TopKV2, new VulkanTopKV2Creator);
    return true;
}();

}