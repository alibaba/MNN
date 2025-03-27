//
//  VulkanTexture.cpp
//  MNN
//
//  Created by MNN on 2023/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "VulkanBasicExecution.hpp"

namespace MNN {
struct ConstBuffer {
    ivec4 inShape;  // inW, inH, unit, gridunit
    ivec4 outShape; // outW, outH, unit, batch
    bool alignCorners;
};

struct Float2IntBuffer {
    ivec4 size;
    vec4 unit;
};

class VulkanTexture : public VulkanBasicExecution {
public:
    VulkanTexture(SampleMode mode, bool isCube, Backend* bn, bool grad) : VulkanBasicExecution(bn) {
        mIsCube = isCube;
        mGrad = grad;
        auto vkBn = (VulkanBackend*)bn;
        std::vector<VkDescriptorType> types{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        };
        if (isCube) {
            if (mode == SampleMode_NEAREST) {
                if (grad) {
                    mPipeline = vkBn->getPipeline("glsl_texturecubegrad_NEAREST_comp", types);
                } else {
                    mPipeline = vkBn->getPipeline("glsl_texturecube_NEAREST_comp", types);
                }
            } else {
                if (grad) {
                    mPipeline = vkBn->getPipeline("glsl_texturecubegrad_comp", types);
                } else {
                    mPipeline = vkBn->getPipeline("glsl_texturecube_comp", types);
                }
            }
        } else {
            if (mode == SampleMode_NEAREST) {
                if (grad) {
                    mPipeline = vkBn->getPipeline("glsl_texture2dgrad_NEAREST_comp", types);
                } else {
                    mPipeline = vkBn->getPipeline("glsl_texture2d_NEAREST_comp", types);
                }
            } else {
                if (grad) {
                    mPipeline = vkBn->getPipeline("glsl_texture2dgrad_comp", types);
                } else {
                    mPipeline = vkBn->getPipeline("glsl_texture2d_comp", types);
                }
            }
        }
        mConstBuffer = vkBn->allocUniform(nullptr, sizeof(ConstBuffer));
        if (grad) {
            std::vector<VkDescriptorType> types {
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            };
            mIndiceCopyPipeline = vkBn->getPipeline("glsl_int2float_comp", types);
            mIndiceCopySet.reset(mIndiceCopyPipeline->createSet());
            mIndiceConstBuffer = vkBn->allocUniform(nullptr, sizeof(Float2IntBuffer));
        }
        mDescriptorSet.reset(mPipeline->createSet());
    }
    virtual ~VulkanTexture() {
        auto vkBn = (VulkanBackend*)backend();
        vkBn->recycleUniform(mConstBuffer);
        if (nullptr != mIndiceConstBuffer) {
            vkBn->recycleUniform(mIndiceConstBuffer);
        }
    }
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override {
        Tensor* inputTensor;
        Tensor* gridTensor;
        Tensor* outputTensor;
        if (mGrad) {
            inputTensor = outputs[0];
            gridTensor = inputs[1];
            outputTensor = inputs[0];
        } else {
            inputTensor = inputs[0];
            gridTensor = inputs[1];
            outputTensor = outputs[0];
        }
        auto batches = inputTensor->length(0);
        auto unit = outputTensor->length(3);
        int ih, iw;
        if (mIsCube) {
            ih = inputTensor->length(2);
            iw = inputTensor->length(3);
        } else {
            ih = inputTensor->length(1);
            iw = inputTensor->length(2);
        }
        auto oh = outputTensor->length(1);
        auto ow = outputTensor->length(2);
        // gpu param
        {
            auto parm = reinterpret_cast<ConstBuffer*>(mConstBuffer->map());
            parm->inShape[0] = iw;
            parm->inShape[1] = ih;
            parm->inShape[2] = unit;
            parm->inShape[3] = gridTensor->length(3);
            parm->outShape[0] = ow;
            parm->outShape[1] = oh;
            parm->outShape[2] = unit;
            parm->outShape[3] = batches;
            parm->alignCorners = false;
            mConstBuffer->unmap();
        }
        auto vkBn = static_cast<VulkanBackend*>(backend());
        mDescriptorSet->writeBuffer(vkBn->getBuffer(outputTensor), 0);
        mDescriptorSet->writeBuffer(vkBn->getBuffer(gridTensor), 2);
        mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 3, mConstBuffer->size());
        MemChunk tempMem;
        VulkanBuffer* midBuffer = nullptr;
        if (mGrad) {
            auto memalloc = vkBn->getDynamicMemoryPool();
            tempMem = memalloc->alloc(inputTensor->size());
            if (tempMem.first == nullptr) {
                return OUT_OF_MEMORY;
            }
            midBuffer = (VulkanBuffer*)tempMem.first;
            mDescriptorSet->writeBuffer(midBuffer->buffer(), 1, midBuffer->size(), tempMem.second);
            memalloc->free(tempMem);
            vkCmdFillBuffer(cmdBuffer->get(), midBuffer->buffer(), tempMem.second, midBuffer->size(), 0);
            cmdBuffer->barrierSource(midBuffer->buffer(), tempMem.second, midBuffer->size(), VulkanCommandPool::Buffer::WRITE_WRITE);
        } else {
            mDescriptorSet->writeBuffer(vkBn->getBuffer(inputTensor), 1);
        }
        mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(ow * oh * unit * batches, 256), 1, 1);
        if (mGrad) {
            cmdBuffer->barrierSource(midBuffer->buffer(), tempMem.second, midBuffer->size(), VulkanCommandPool::Buffer::READ_WRITE);
            auto param = reinterpret_cast<Float2IntBuffer*>(mIndiceConstBuffer->map());
            auto totalInputSize = iw * ih * unit * batches;
            if (mIsCube) {
                totalInputSize = totalInputSize * 6;
            }
            param->size[0] = totalInputSize;
            param->size[1] = 1;
            param->size[2] = 1;
            param->size[3] = 1;
            param->unit[0] = 1.0f / 16777216.0f;
            param->unit[1] = 0.0f;
            param->unit[2] = 0.0f;
            param->unit[3] = 0.0f;
            mIndiceConstBuffer->unmap();
            mIndiceCopySet->writeBuffer(vkBn->getBuffer(inputTensor), 0);
            mIndiceCopySet->writeBuffer(midBuffer->buffer(), 1, midBuffer->size(), tempMem.second);
            mIndiceCopySet->writeBuffer(mIndiceConstBuffer->buffer(), 2, mIndiceConstBuffer->size());
            mIndiceCopyPipeline->bind(cmdBuffer->get(), mIndiceCopySet->get());
            vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalInputSize, 256), 1, 1);
        }
        return NO_ERROR;
    }

private:
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    const VulkanPipeline* mPipeline;
    const VulkanPipeline* mIndiceCopyPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mIndiceCopySet;
    std::shared_ptr<VulkanBuffer> mIndiceConstBuffer;
    bool mIsCube;
    bool mGrad;
};

class VulkanTextureCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        auto gridSampleParam = op->main_as_GridSample();
        auto mode = gridSampleParam->paddingMode();
        bool isCube = mode == BorderMode_CUBE;
        if (gridSampleParam->backward()) {
            return new VulkanTexture(gridSampleParam->mode(), isCube, backend, true);
        }
        return new VulkanTexture(gridSampleParam->mode(), isCube, backend, false);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Texture, new VulkanTextureCreator);
    return true;
}();

} // namespace MNN

