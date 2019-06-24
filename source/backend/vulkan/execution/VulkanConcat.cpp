//
//  VulkanConcat.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanConcat.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"
namespace MNN {
struct ConcatParam {
    ivec4 inImageSize;
    ivec4 outImageSize;
    ivec4 offset; // w, h, c, 0
};

static void _setGPUParam(VulkanBuffer* paramBuffer, const Tensor* inputShape, Tensor* outputShape, bool imageLayout) {
    auto data = reinterpret_cast<ConcatParam*>(paramBuffer->map());
    ::memset(data, 0, sizeof(ConcatParam));
    data->inImageSize[0]  = inputShape->width();
    data->inImageSize[1]  = inputShape->height();
    data->inImageSize[2]  = UP_DIV(inputShape->channel(), 4);
    data->inImageSize[3]  = inputShape->batch();
    data->outImageSize[0] = outputShape->width();

    paramBuffer->unmap();
}

VulkanConcat::VulkanConcat(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto axis  = op->main_as_Axis()->axis();
    mAxis      = axis;
    mVkbackend = static_cast<VulkanBackend*>(bn);
}

ErrorCode VulkanConcat::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    auto output = outputs[0];

    if (TensorUtils::getDescribe(output)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
        MNN_PRINT("Vulkan Concat NOT SUPPORT for Buffer Layout Now!\n");
        return NOT_SUPPORT;
    }

    int axis = mAxis;
    if (0 > axis) {
        axis = output->dimensions() + axis;
    }
    bool fastMode = true;
    if (1 == axis) {
        for (int i = 0; i < inputs.size() - 1; ++i) {
            auto input = inputs[i];
            if (input->channel() % 4 != 0) {
                fastMode = false;
                break;
            }
        }
    }

    if (fastMode) {
        mImageConcat = std::make_shared<VulkanConcatImageImpl>(axis, mVkbackend);
        mImageConcat->encodeImageImpl(inputs, output, cmdBuffer);
    } else {
        mBufferConcat = std::make_shared<VulkanConcatBufferImpl>(axis, mVkbackend);
        mBufferConcat->encodeBufferImpl(inputs, output, cmdBuffer);
    }

    return NO_ERROR;
}

VulkanConcatImageImpl::VulkanConcatImageImpl(int axis, VulkanBackend* vkBackend) : mAxis(axis), mVkbackend(vkBackend) {
    mSampler = vkBackend->getCommonSampler();
}

ErrorCode VulkanConcatImageImpl::encodeImageImpl(const std::vector<Tensor*>& inputs, Tensor* output,
                                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    mConstBuffers.clear();
    mSets.clear();
    int axisOffset = 0;

    auto pipeline = mVkbackend->getPipeline(
        "glsl_blitC4_comp", /*glsl_blitC4_comp, glsl_blitC4_comp_len,*/ std::vector<VkDescriptorType>{
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER});
    for (int i = 0; i < inputs.size(); ++i) {
        auto input       = inputs[i];
        int icDiv4       = UP_DIV(input->channel(), 4);
        auto constBuffer = std::make_shared<VulkanBuffer>(mVkbackend->getMemoryPool(), false, sizeof(ConcatParam),
                                                          nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        mConstBuffers.push_back(constBuffer);
        auto constValue = reinterpret_cast<ConcatParam*>(constBuffer->map());
        ::memset(constValue, 0, sizeof(ConcatParam));
        constValue->inImageSize[0]  = input->width();
        constValue->inImageSize[1]  = input->height();
        constValue->inImageSize[2]  = icDiv4;
        constValue->inImageSize[3]  = input->batch();
        constValue->outImageSize[0] = output->width();
        constValue->outImageSize[1] = output->height();
        constValue->outImageSize[2] = UP_DIV(output->channel(), 4);
        constValue->outImageSize[3] = output->batch();
        switch (mAxis) {
            case 0:
                constValue->offset[2] = axisOffset;
                axisOffset += input->batch() * icDiv4;
                break;
            case 1:
                constValue->offset[2] = axisOffset;
                axisOffset += icDiv4;
                break;
            case 2:
                constValue->offset[1] = axisOffset;
                axisOffset += input->height();
                break;
            case 3:
                constValue->offset[0] = axisOffset;
                axisOffset += input->width();
                break;
            default:
                return NOT_SUPPORT;
        }
        constBuffer->unmap();
        std::shared_ptr<VulkanPipeline::DescriptorSet> desSet;
        desSet.reset(pipeline->createSet());
        desSet->writeImage(reinterpret_cast<VkImageView>(output->deviceId()), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL,
                           0);
        desSet->writeImage(reinterpret_cast<VkImageView>(input->deviceId()), mSampler->get(),
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        desSet->writeBuffer(constBuffer->buffer(), 2, constBuffer->size());
        pipeline->bind(cmdBuffer->get(), desSet->get());
        mSets.push_back(desSet);
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(input->width(), 16), UP_DIV(input->height(), 16),
                      icDiv4 * input->batch());
    }

    return NO_ERROR;
}

VulkanConcatBufferImpl::VulkanConcatBufferImpl(int axis, VulkanBackend* vkBackend)
    : mAxis(axis), mVkbackend(vkBackend) {
}

ErrorCode VulkanConcatBufferImpl::encodeBufferImpl(const std::vector<Tensor*>& inputs, Tensor* output,
                                                   const VulkanCommandPool::Buffer* cmdBuffer) {
    const int inputSize = inputs.size();
    // set temp-output tensor layout and acquire memory for temp-output tensor
    mTempOutputTensor = std::make_shared<Tensor>(4);
    TensorUtils::copyShape(output, mTempOutputTensor.get());
    TensorUtils::getDescribe(mTempOutputTensor.get())->dimensionFormat = MNN_DATA_FORMAT_NCHW;
    mTempOutputTensor->buffer().dim[1].flags                           = 0;
    mVkbackend->onAcquireBuffer(mTempOutputTensor.get(), Backend::DYNAMIC);
    // set temp-input tensors layout and acquire memory for temp-input tensors
    mTempInputTensors.clear();
    for (int i = 0; i < inputSize; ++i) {
        auto inputTemp = std::make_shared<Tensor>();
        TensorUtils::copyShape(inputs[i], inputTemp.get());
        TensorUtils::getDescribe(inputTemp.get())->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        inputTemp->buffer().dim[1].flags                           = 0;
        mTempInputTensors.push_back(inputTemp);
        mVkbackend->onAcquireBuffer(inputTemp.get(), Backend::DYNAMIC);
    }

    // reset converter
    // image to nchw
    for (int i = 0; i < inputSize; ++i) {
        auto converter = std::make_shared<VulkanImageConverter>(mVkbackend);
        mTensorConvert4Inputs.push_back(converter);
    }
    // nchw to image
    mTensorConvert4Output = std::make_shared<VulkanImageConverter>(mVkbackend);

    // encode
    for (int i = 0; i < inputSize; ++i) {
        mTensorConvert4Inputs[i]->encodeTensorToBuffer(
            inputs[i], reinterpret_cast<VkBuffer>(mTempInputTensors[i]->deviceId()), mTempInputTensors[i]->size(), 0,
            MNN_DATA_FORMAT_NCHW, cmdBuffer);
    }
    // concat
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    auto bufferConcatPipeline = mVkbackend->getPipeline("glsl_concatBuffer_comp",
                                                        /*glsl_concatBuffer_comp, glsl_concatBuffer_comp_len,*/ types);
    int axisOffset            = 0;
    for (int i = 0; i < inputSize; ++i) {
        auto& tempInput  = mTempInputTensors[i];
        auto constBuffer = std::make_shared<VulkanBuffer>(mVkbackend->getMemoryPool(), false, sizeof(ConcatParam),
                                                          nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        mConstBuffers.push_back(constBuffer);
        auto dataPtr = reinterpret_cast<ConcatParam*>(constBuffer->map());
        ::memset(dataPtr, 0, sizeof(ConcatParam));
        dataPtr->inImageSize[0]  = tempInput->width();
        dataPtr->inImageSize[1]  = tempInput->height();
        dataPtr->inImageSize[2]  = tempInput->channel();
        dataPtr->inImageSize[3]  = tempInput->batch();
        dataPtr->outImageSize[0] = output->width();
        dataPtr->outImageSize[1] = output->height();
        dataPtr->outImageSize[2] = output->channel();
        dataPtr->outImageSize[3] = output->batch();
        switch (mAxis) {
            case 0:
                dataPtr->offset[2] = axisOffset;
                axisOffset += tempInput->batch() * tempInput->channel();
                break;
            case 1:
                dataPtr->offset[2] = axisOffset;
                axisOffset += tempInput->channel();
                break;
            case 2:
                dataPtr->offset[1] = axisOffset;
                axisOffset += tempInput->height();
                break;
            case 3:
                dataPtr->offset[0] = axisOffset;
                axisOffset += tempInput->width();
                break;

            default:
                return NOT_SUPPORT;
                break;
        }
        constBuffer->unmap();
        std::shared_ptr<VulkanPipeline::DescriptorSet> desSet;
        desSet.reset(bufferConcatPipeline->createSet());
        desSet->writeBuffer(reinterpret_cast<VkBuffer>(mTempOutputTensor->deviceId()), 0, mTempOutputTensor->size());
        desSet->writeBuffer(reinterpret_cast<VkBuffer>(tempInput->deviceId()), 1, tempInput->size());
        desSet->writeBuffer(constBuffer->buffer(), 2, constBuffer->size());
        bufferConcatPipeline->bind(cmdBuffer->get(), desSet->get());
        mSets.push_back(desSet);
        cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(tempInput->deviceId()), 0, tempInput->size());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(tempInput->width(), 16), UP_DIV(tempInput->height(), 16),
                      tempInput->channel() * tempInput->batch());
    }
    // back to image for temp-output tensor
    mTensorConvert4Output->encodeBufferToTensor(reinterpret_cast<VkBuffer>(mTempOutputTensor->deviceId()), output,
                                                mTempOutputTensor->size(), 0, MNN_DATA_FORMAT_NCHW, cmdBuffer);

    // reuse memory
    mVkbackend->onReleaseBuffer(mTempOutputTensor.get(), Backend::DYNAMIC);
    for (auto& item : mTempInputTensors) {
        mVkbackend->onReleaseBuffer(item.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

class VulkanConcatCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new VulkanConcat(op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Concat, new VulkanConcatCreator);
    return true;
}();

} // namespace MNN
