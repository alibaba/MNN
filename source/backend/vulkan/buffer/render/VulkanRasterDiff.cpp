
//
//  VulkanRasterDiff.cpp
//  MNN
//
//  Created by MNN on 2023/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "VulkanBasicExecution.hpp"

namespace MNN {
struct ConstBuffer {
    ivec4 inShape;  // inW, inH
};

class VulkanRasterDiff : public VulkanBasicExecution {
public:
    VulkanRasterDiff(Backend* bn, bool grad) : VulkanBasicExecution(bn) {
        auto vkBn = (VulkanBackend*)bn;
        std::vector<VkDescriptorType> types{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        };
        if (grad) {
            mPipeline = vkBn->getPipeline("glsl_dfdxdygrad_comp", types);
        } else {
            mPipeline = vkBn->getPipeline("glsl_dfdxdy_comp", types);
        }
        mConstBuffer = vkBn->allocUniform(nullptr, sizeof(ConstBuffer));
        mDescriptorSet.reset(mPipeline->createSet());
        mGrad = grad;
    }
    virtual ~VulkanRasterDiff() {
        auto vkBn = (VulkanBackend*)backend();
        vkBn->recycleUniform(mConstBuffer);
    }
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override {
        Tensor* inputTensor;
        Tensor* dxTensor;
        Tensor* dyTensor;
        if (!mGrad) {
            inputTensor = inputs[0];
            dxTensor = outputs[0];
            dyTensor = outputs[1];
        } else {
            inputTensor = outputs[0];
            dxTensor = inputs[0];
            dyTensor = inputs[1];
        }
        auto batches = inputTensor->length(0);
        auto unit = inputTensor->length(3);
        int iw = inputTensor->length(1);
        int ih = inputTensor->length(2);
        // gpu param
        {
            auto parm = reinterpret_cast<ConstBuffer*>(mConstBuffer->map());
            parm->inShape[0] = iw;
            parm->inShape[1] = ih;
            parm->inShape[2] = unit;
            parm->inShape[3] = batches;
            mConstBuffer->unmap();
        }
        auto vkBn = static_cast<VulkanBackend*>(backend());
        mDescriptorSet->writeBuffer(vkBn->getBuffer(dxTensor), 0);
        mDescriptorSet->writeBuffer(vkBn->getBuffer(dyTensor), 1);
        mDescriptorSet->writeBuffer(vkBn->getBuffer(inputTensor), 2);
        mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 3, mConstBuffer->size());
        mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(iw * ih * unit * batches, 256), 1, 1);
        return NO_ERROR;
    }
private:
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    const VulkanPipeline* mPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    bool mGrad;
};

class VulkanRasterDiffCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        auto extra = op->main_as_Extra();
        if (nullptr != extra) {
            // Grad
            return new VulkanRasterDiff(backend, true);
        }
        return new VulkanRasterDiff(backend, false);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_RasterDiff, new VulkanRasterDiffCreator);
    return true;
}();

} // namespace MNN

