//
//  VulkanSoftmax.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanSoftmax.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

struct ConstBuffer {
    int w;
    int h;
    int c;
};

VulkanSoftmax::VulkanSoftmax(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    const auto softmaxParam = op->main_as_Axis();
    mAxis                   = softmaxParam->axis();
    auto vkBn = (VulkanBackend*)backend();
    mConstBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, sizeof(ConstBuffer), nullptr,
                                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    };
    mSoftmaxPipeline =
        vkBn->getPipeline("glsl_softmaxHeight_NHWC_comp", types);
    mDescriptorSet.reset(mSoftmaxPipeline->createSet());
}

VulkanSoftmax::~VulkanSoftmax() {
}

ErrorCode VulkanSoftmax::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto extra = (VulkanBackend*)backend();
    auto inputFormat = TensorUtils::getDescribe(input)->dimensionFormat;
    auto axis = mAxis;
    if (axis < 0) {
        axis = input->dimensions() + axis;
    }
    if (inputFormat == MNN_DATA_FORMAT_NC4HW4) {
        mConvert.reset(new ConvertComponent);
        mConvert->mTempInputTensor.reset(new Tensor(input, Tensor::CAFFE, false));
        mConvert->mTempOutputTensor.reset(new Tensor(output, Tensor::CAFFE, false));
        auto res = backend()->onAcquireBuffer(mConvert->mTempInputTensor.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        res = backend()->onAcquireBuffer(mConvert->mTempOutputTensor.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        mConvert->mInputConvert = VulkanRaster::create(mConvert->mTempInputTensor.get(), backend());
        mConvert->mOutputConvert = VulkanRaster::create(mConvert->mTempOutputTensor.get(), backend());
        TensorUtils::getDescribe(mConvert->mTempInputTensor.get())->regions = {TensorUtils::makeFullSlice(inputs[0])};
        TensorUtils::getDescribe(outputs[0])->regions = {TensorUtils::makeFullSlice(mConvert->mTempOutputTensor.get())};
        input = mConvert->mTempInputTensor.get();
        output = mConvert->mTempOutputTensor.get();
        
        mConvert->mInputConvert.exe->onEncode({}, {mConvert->mTempInputTensor.get()}, cmdBuffer);
        auto inputBuffer = extra->getTensorBuffer(input);
        auto inputBufferSize = extra->getTensorSize(input);
        cmdBuffer->barrierSource(inputBuffer.first->buffer(), inputBuffer.second, inputBufferSize);
    }
    int inside = 1;
    int outside = 1;
    int mid = input->length(axis);
    for (int i=0; i<axis; ++i) {
        outside *= input->length(i);
    }
    for (int i=axis+1; i<output->dimensions(); ++i) {
        inside *= input->length(i);
    }
    // gpu param
    {
        auto softmax = reinterpret_cast<ConstBuffer*>(mConstBuffer->map());
        ::memset(softmax, 0, sizeof(ConstBuffer));
        softmax->w = inside;
        softmax->h = mid;
        softmax->c = outside;
        mConstBuffer->unmap();
    }
    auto outputBuffer = extra->getTensorBuffer(output);
    auto inputBuffer = extra->getTensorBuffer(input);
    mDescriptorSet->writeBuffer(outputBuffer.first->buffer(), 0, extra->getTensorSize(output), outputBuffer.second);
    mDescriptorSet->writeBuffer(inputBuffer.first->buffer(), 1, extra->getTensorSize(input), inputBuffer.second);
    mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 2, mConstBuffer->size());
    mSoftmaxPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(outside, 8), UP_DIV(inside, 8), 1);
    if (inputFormat == MNN_DATA_FORMAT_NC4HW4) {
        cmdBuffer->barrierSource(outputBuffer.first->buffer(), outputBuffer.second, extra->getTensorSize(output));
        mConvert->mOutputConvert.exe->onEncode({}, {outputs[0]}, cmdBuffer);
        backend()->onReleaseBuffer(mConvert->mTempInputTensor.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mConvert->mTempOutputTensor.get(), Backend::DYNAMIC);
    }

    return NO_ERROR;
}

class VulkanSoftmaxCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new VulkanSoftmax(op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Softmax, new VulkanSoftmaxCreator);
    return true;
}();

} // namespace MNN
