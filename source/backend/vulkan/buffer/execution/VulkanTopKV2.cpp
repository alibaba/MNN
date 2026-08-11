//
//  VulkanTopKV2.cpp
//  MNN
//
//  Vulkan buffer-mode implementation of TopKV2.
//

#include "VulkanTopKV2.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

struct TopKV2ConstBuffer {
    int rowSize;
    int k;
    int numRows;
    int pad;
};

VulkanTopKV2::VulkanTopKV2(const Op* op, Backend* bn, int k, Tensor* input) : VulkanBasicExecution(bn) {
    MNN_ASSERT(k > 0);
    MNN_ASSERT(input != nullptr);
    auto vkBn = (VulkanBackend*)backend();
    mK = k;
    mLargest = true;
    auto param = op->main_as_TopKV2();
    if (nullptr != param) {
        mLargest = param->largest();
    }

    mConstBuffer = vkBn->allocUniform();
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };

    std::string pKey = "glsl_topkv2_";
    if (mLargest) {
        pKey += "SORT_DESC_";
    }
    if (input->getType().code == halide_type_float && vkBn->useFP16()) {
        pKey += "FP16_";
    }
    pKey += "comp";

    mPipeline = vkBn->getPipeline(pKey, types);
    mDescriptorSet.reset(mPipeline->createSet());
}

VulkanTopKV2::~VulkanTopKV2() {
    auto vkBn = (VulkanBackend*)backend();
    vkBn->recycleUniform(mConstBuffer);
}

ErrorCode VulkanTopKV2::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input = inputs[0];
    auto outputValue = outputs[0];
    auto outputIndex = outputs[1];

    const int rowSize = input->length(input->dimensions() - 1);
    if (rowSize <= 0) {
        return NO_ERROR;
    }
    const int numRows = input->elementSize() / rowSize;
    const int k = mK;

    auto vkBn = static_cast<VulkanBackend*>(backend());

    // Set GPU params
    auto topkParam = reinterpret_cast<TopKV2ConstBuffer*>(mConstBuffer->map());
    topkParam->rowSize = rowSize;
    topkParam->k = k;
    topkParam->numRows = numRows;
    topkParam->pad = 0;
    mConstBuffer->unmap();

    // Bind buffers
    mDescriptorSet->writeBuffer(vkBn->getBuffer(outputValue), 0);
    mDescriptorSet->writeBuffer(vkBn->getBuffer(outputIndex), 1);
    mDescriptorSet->writeBuffer(vkBn->getBuffer(input), 2);
    mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 3, mConstBuffer->size());

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
        return new VulkanTopKV2(op, backend, k, inputs[0]);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_TopKV2, new VulkanTopKV2Creator);
    return true;
}();

} // namespace MNN
