//
//  VulkanPool.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanPool.hpp"
#include "core/Macro.h"
namespace MNN {
struct ConstBuffer {
    ivec4 inputSize;
    ivec4 outputSize;
    ivec2 pad;
    ivec2 kernelSize;
    ivec2 stride;
    ivec2 count;
};

VulkanPool::VulkanPool(const Op* op, Backend* bn, Tensor * tensor) : VulkanBasicExecution(bn) {
    mCommon    = op->main_as_Pool();
    auto extra = static_cast<VulkanBackend*>(bn);

    std::string pKey;
    auto poolType = mCommon->type();
    MNN_ASSERT(poolType == PoolType_MAXPOOL || poolType == PoolType_AVEPOOL);
    if (poolType == PoolType_MAXPOOL) {
        pKey = "glsl_maxpool_";
    }
    if (poolType == PoolType_AVEPOOL) {
        pKey = "glsl_avgpool_";
    }
    if (tensor->getType().code ==halide_type_float && extra->useFP16()) {
        pKey += "FP16_";
    }
    pKey += "comp";
    std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};

    mPoolPipeline = extra->getPipeline(pKey, types);
    mConstBuffer = extra->allocUniform(nullptr, sizeof(ConstBuffer));
    mDescriptorSet.reset(mPoolPipeline->createSet());
}
VulkanPool::~VulkanPool() {
    auto extra = static_cast<VulkanBackend*>(backend());
    extra->recycleUniform(mConstBuffer);
}

ErrorCode VulkanPool::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];
    int iw      = input->width();
    int ih      = input->height();
    int ow      = output->width();
    int oh      = output->height();
    int icDiv4  = UP_DIV(input->channel(), 4);
    int ocDiv4  = UP_DIV(output->channel(), 4);
    auto extra  = (VulkanBackend*)backend();
    // Set Const Buffer
    {
        auto pool = (ConstBuffer*)mConstBuffer->map();
        ::memset(pool, 0, sizeof(ConstBuffer));
        pool->inputSize[0]  = input->width();
        pool->inputSize[1]  = input->height();
        pool->inputSize[2]  = icDiv4 * input->batch();
        pool->outputSize[0] = ow;
        pool->outputSize[1] = oh;
        pool->outputSize[2] = ocDiv4 * output->batch();
        int padWidth     = mCommon->padX();
        int padHeight    = mCommon->padY();

        int strideWidth  = mCommon->strideX();
        int strideHeight = mCommon->strideY();

        // edit const if global
        int kernelWidth  = std::min(mCommon->kernelX(), iw);
        int kernelHeight = std::min(mCommon->kernelY(), ih);
        if (mCommon->isGlobal()) {
            kernelWidth  = iw;
            kernelHeight = ih;
            strideWidth  = iw;
            strideHeight = ih;
            padWidth     = 0;
            padHeight    = 0;
        }

        if(mCommon->padType() == PoolPadType_SAME){
            int padNeededWidth  = (output->width() - 1) * strideWidth + kernelWidth - input->width();
            int padNeededHeight = (output->height() - 1) * strideHeight + kernelHeight - input->height();
            padWidth            = padNeededWidth > 0 ? padNeededWidth / 2 : 0;
            padHeight           = padNeededHeight > 0 ? padNeededHeight / 2 : 0;
        } else if (mCommon->padType() == PoolPadType_VALID) {
            padWidth = padHeight = 0;
        }

        pool->pad[0]        = padWidth;
        pool->pad[1]        = padHeight;
        pool->stride[0]     = strideWidth;
        pool->stride[1]     = strideHeight;
        pool->kernelSize[0] = kernelWidth;
        pool->kernelSize[1] = kernelHeight;

        auto countType = mCommon->countType();
        if (countType == AvgPoolCountType_DEFAULT) {
            if (mCommon->padType() == PoolPadType_CAFFE) {
                countType = AvgPoolCountType_INCLUDE_PADDING;
            } else {
                countType = AvgPoolCountType_EXCLUDE_PADDING;
            }
        }
        pool->count[0] = (countType == AvgPoolCountType_INCLUDE_PADDING) ? 1 : 0;
        pool->count[1] = 0;

        mConstBuffer->unmap();
    }

    // Set Command Buffer
    {
        auto outputT = extra->getBuffer(output);
        auto inputT = extra->getBuffer(input);
        mDescriptorSet->writeBuffer(outputT, 0);
        mDescriptorSet->writeBuffer(inputT, 1);
        mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 2, mConstBuffer->size());
        mPoolPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(ow, 8), UP_DIV(oh, 8), UP_DIV(ocDiv4 * output->batch(), 1));
    }
    return NO_ERROR;
}

class VulkanPoolCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new VulkanPool(op, backend, outputs[0]);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Pooling, new VulkanPoolCreator);
    return true;
}();

} // namespace MNN
