//
//  VulkanLSTM.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanLSTM.hpp"
#include "Macro.h"

namespace MNN {

struct GpuParam {
    ivec4 shape;
};
static void _setGPUParam(std::shared_ptr<VulkanBuffer>& paramBuffer, int i1, int i2, int i3, int i4) {
    auto VulkanLSTMParam = reinterpret_cast<GpuParam*>(paramBuffer->map());
    ::memset(VulkanLSTMParam, 0, sizeof(GpuParam));
    VulkanLSTMParam->shape[0] = i1;
    VulkanLSTMParam->shape[1] = i2;
    VulkanLSTMParam->shape[2] = i3;
    VulkanLSTMParam->shape[3] = i4;
    paramBuffer->flush(true, 0, sizeof(GpuParam));
    paramBuffer->unmap();
}

LSTMChannel::LSTMChannel(const VulkanPipeline* vulkanLSTMPipeline, VulkanBackend* vkbackend, const int channel)
    : mChannel(channel), mVulkanLSTMPipeline(vulkanLSTMPipeline) {
    mParamBuffer.reset(new VulkanBuffer(vkbackend->getMemoryPool(), false, sizeof(GpuParam), nullptr,
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
}
LSTMChannel::~LSTMChannel() {
}

ErrorCode LSTMChannel::encodeImpl(std::shared_ptr<VulkanBuffer>& gates, std::shared_ptr<VulkanBuffer>& cells,
                                  std::shared_ptr<VulkanBuffer>& weightH, std::shared_ptr<VulkanBuffer>& bias,
                                  std::shared_ptr<VulkanBuffer>& out, const VulkanCommandPool::Buffer* cmdBuffer,
                                  const int ow) {
    //        const int gateOffset = (mChannel / 4) * ow * 16 + mChannel % 4;

    _setGPUParam(mParamBuffer, ow, mChannel, 0, 0);

    mDescriptorSet.reset(mVulkanLSTMPipeline->createSet());
    mDescriptorSet->writeBuffer(gates->buffer(), 0, gates->size(), 0);
    mDescriptorSet->writeBuffer(cells->buffer(), 1, cells->size(), 0);
    mDescriptorSet->writeBuffer(weightH->buffer(), 2, weightH->size(), 0);
    mDescriptorSet->writeBuffer(bias->buffer(), 3, bias->size(), 0);
    mDescriptorSet->writeBuffer(out->buffer(), 4, out->size(), 0);
    mDescriptorSet->writeBuffer(mParamBuffer->buffer(), 5, mParamBuffer->size(), 0);
    mVulkanLSTMPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    cmdBuffer->barrierSource(gates->buffer(), 0, gates->size());

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(ow, 8), 1, 1);

    return NO_ERROR;
}

VulkanLSTM::VulkanLSTM(const LSTM* lstm, Backend* bn) : VulkanBasicExecution(bn), mLSTM(lstm) {
    std::vector<VkDescriptorType> VulkanLSTMTypes{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };

    std::vector<VkDescriptorType> VulkanLSTMGateTypes{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };

    std::vector<VkDescriptorType> VulkanLSTMSaveTypes{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };

    mVKbackend = static_cast<VulkanBackend*>(bn);
    mVulkanLSTMPipeline =
        mVKbackend->getPipeline("glsl_lstm_comp", /*glsl_lstm_comp, glsl_lstm_comp_len,*/ VulkanLSTMTypes);
    mVulkanLSTMGatePipeline = mVKbackend->getPipeline(
        "glsl_lstmGate_comp", /*glsl_lstmGate_comp, glsl_lstmGate_comp_len,*/ VulkanLSTMGateTypes);
    mVulkanLSTMSavePipeline = mVKbackend->getPipeline(
        "glsl_lstmSave_comp", /*glsl_lstmSave_comp, glsl_lstmSave_comp_len,*/ VulkanLSTMSaveTypes);

    mGateParamBuffer.reset(new VulkanBuffer(mVKbackend->getMemoryPool(), false, sizeof(GpuParam), nullptr,
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    mSaveParamBuffer.reset(new VulkanBuffer(mVKbackend->getMemoryPool(), false, sizeof(GpuParam), nullptr,
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
}
VulkanLSTM::~VulkanLSTM() {
}

ErrorCode VulkanLSTM::_resize(const Tensor* input, const Tensor* output) {
    const int iw = input->width();
    const int ow = output->width();
    auto weightI = mLSTM->weightI();
    auto weightH = mLSTM->weightH();
    auto bias    = mLSTM->bias();

    const int weightSize = weightI->dims()->data()[0];
    bool devided         = weightI && !weightH && weightSize == (4 * ow * (iw + ow + 2));

    if (devided) {
        auto data = weightI->float32s()->data();
        mWeightI.reset(new VulkanBuffer(mVKbackend->getMemoryPool(), false, sizeof(float) * ow * iw * 4, nullptr,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        {
            auto weightIPtr = reinterpret_cast<float*>(mWeightI->map());
            const int step  = iw * ow;
            ::memcpy(weightIPtr, data, 2 * step * sizeof(float));
            weightIPtr += 2 * step;
            data += 2 * step;
            ::memcpy(weightIPtr, data + step, step * sizeof(float));
            ::memcpy(weightIPtr + step, data, step * sizeof(float));
            mWeightI->flush(true, 0, mWeightI->size());
            mWeightI->unmap();
            data += 2 * step;
        }
        mWeightH.reset(new VulkanBuffer(mVKbackend->getMemoryPool(), false, sizeof(float) * ow * ow * 4, nullptr,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        {
            auto weightHPtr = reinterpret_cast<float*>(mWeightH->map());
            const int step  = ow * ow;
            for (int i = 0; i < step; ++i, weightHPtr += 4) {
                weightHPtr[0] = data[i];
                weightHPtr[1] = data[i + step];
                weightHPtr[2] = data[i + 3 * step];
                weightHPtr[3] = data[i + 2 * step];
            }
            mWeightH->flush(true, 0, mWeightH->size());
            mWeightH->unmap();
            data += 4 * step;
        }
        mBias.reset(new VulkanBuffer(mVKbackend->getMemoryPool(), false, sizeof(float) * ow * 4, nullptr,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        {
            auto biasPtr   = reinterpret_cast<float*>(mBias->map());
            auto from      = data;
            const int step = ow;
            for (int i = 0; i < step; ++i, biasPtr += 4, from++) {
                biasPtr[0] = from[0];
                biasPtr[1] = from[step];
                biasPtr[2] = from[3 * step];
                biasPtr[3] = from[2 * step];
            }
            mBias->unmap();
        }

    } else {
        mWeightI.reset(new VulkanBuffer(mVKbackend->getMemoryPool(), false, sizeof(float) * ow * iw * 4,
                                        weightI->float32s()->data(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        mWeightH.reset(new VulkanBuffer(mVKbackend->getMemoryPool(), false, sizeof(float) * ow * ow * 4, nullptr,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        {
            auto weightHPtr = reinterpret_cast<float*>(mWeightH->map());
            auto from       = weightH->float32s()->data();
            const int sect  = ow * ow;
            for (int i = 0; i < sect; ++i, weightHPtr += 4, from++) {
                weightHPtr[0] = from[0];
                weightHPtr[1] = from[sect];
                weightHPtr[2] = from[2 * sect];
                weightHPtr[3] = from[3 * sect];
            }
            mWeightH->flush(true, 0, mWeightH->size());
            mWeightH->unmap();
        }

        mBias.reset(new VulkanBuffer(mVKbackend->getMemoryPool(), false, sizeof(float) * ow * 4, nullptr,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        {
            auto biasPtr   = reinterpret_cast<float*>(mBias->map());
            auto from      = bias->float32s()->data();
            const int sect = ow;
            for (int i = 0; i < sect; ++i, biasPtr += 4, from++) {
                biasPtr[0] = from[0];
                biasPtr[1] = from[sect];
                biasPtr[2] = from[2 * sect];
                biasPtr[3] = from[3 * sect];
            }
            mBias->unmap();
        }
    }

    return NO_ERROR;
}

ErrorCode VulkanLSTM::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input       = inputs[0];
    auto output      = outputs[0];
    const int ic     = input->channel();
    const int icDiv4 = UP_DIV(ic, 4);
    const int iw     = input->width();
    const int ow     = output->width();
    _resize(input, output);

    // acquire buffer
    mGate.reset(new VulkanBuffer(mVKbackend->getDynamicMemoryPool(), false, sizeof(float) * 4 * ow * icDiv4 * 4,
                                 nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    mCell.reset(new VulkanBuffer(mVKbackend->getDynamicMemoryPool(), false, sizeof(float) * ow, nullptr,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    mOutputTemp.reset(new VulkanBuffer(mVKbackend->getDynamicMemoryPool(), false, sizeof(float) * ow * icDiv4 * 4,
                                       nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));

    auto cellPtr = reinterpret_cast<float*>(mCell->map());
    ::memset(cellPtr, 0, mCell->size());
    mCell->unmap();

    mLSTMChannels.resize(ic);
    for (int i = 0; i < ic; ++i) {
        std::shared_ptr<LSTMChannel> subLSTM(new LSTMChannel(mVulkanLSTMPipeline, mVKbackend, i));
        mLSTMChannels[i] = subLSTM;
    }

    auto extra = static_cast<VulkanBackend*>(backend());
    // gate
    _setGPUParam(mGateParamBuffer, ow, iw, icDiv4, 0);
    mGateDescriptorSet.reset(mVulkanLSTMGatePipeline->createSet());
    mGateDescriptorSet->writeBuffer(mGate->buffer(), 0, mGate->size());
    mGateDescriptorSet->writeImage(reinterpret_cast<VkImageView>(input->deviceId()), extra->getCommonSampler()->get(),
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mGateDescriptorSet->writeBuffer(mWeightI->buffer(), 2, mWeightI->size());
    mGateDescriptorSet->writeBuffer(mGateParamBuffer->buffer(), 3, mGateParamBuffer->size());
    mVulkanLSTMGatePipeline->bind(cmdBuffer->get(), mGateDescriptorSet->get());

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(ow, 8), 1, icDiv4);

    // channel
    for (int i = 0; i < ic; ++i) {
        mLSTMChannels[i]->encodeImpl(mGate, mCell, mWeightH, mBias, mOutputTemp, cmdBuffer, ow);
    }
    // nchw -> hc4hw4
    _setGPUParam(mSaveParamBuffer, ow, ic, 0, 0);
    mSaveDescriptorSet.reset(mVulkanLSTMSavePipeline->createSet());
    mSaveDescriptorSet->writeBuffer(mOutputTemp->buffer(), 0, mOutputTemp->size());
    mSaveDescriptorSet->writeImage(reinterpret_cast<VkImageView>(output->deviceId()), extra->getCommonSampler()->get(),
                                   VK_IMAGE_LAYOUT_GENERAL, 1);
    mSaveDescriptorSet->writeBuffer(mSaveParamBuffer->buffer(), 2, mSaveParamBuffer->size());
    mVulkanLSTMSavePipeline->bind(cmdBuffer->get(), mSaveDescriptorSet->get());
    cmdBuffer->barrierSource(mOutputTemp->buffer(), 0, mOutputTemp->size());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(ow, 16), 1, icDiv4);

    mGate->release();
    mCell->release();
    mOutputTemp->release();

    return NO_ERROR;
}

class VulkanLSTMCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanLSTM(op->main_as_LSTM(), bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_LSTM, new VulkanLSTMCreator);
    return true;
}();

} // namespace MNN
