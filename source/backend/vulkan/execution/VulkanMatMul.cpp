//
//  VulkanMatMul.cpp
//  MNN
//
//  Created by MNN on 2020/03/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanMatMul.hpp"
namespace MNN {
VulkanMatMul::Reorder::Reorder(const VulkanBackend* bn, bool transpose, bool revert) {
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    };
    if (revert) {
        mFirst = bn->getPipeline("glsl_nc4hw4Tonchw_comp", types);
    } else {
        mFirst = bn->getPipeline("glsl_nchwTonc4hw4_comp", types);
    }
    mBufferBufferSet.reset(mFirst->createSet());
    mBackend = bn;
    mUnitBuffer.reset(new VulkanBuffer(bn->getMemoryPool(), false, sizeof(nchwBuffer), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    std::string imageShaderName = "glsl_packAsImage4x4";
    std::vector<VkDescriptorType> secondTypes{
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    };

    if (revert) {
        secondTypes[0] = VK_DESCRIPTOR_TYPE_SAMPLER;
        imageShaderName = "glsl_unPackImage4x4";
    }
    if (transpose) {
        imageShaderName = imageShaderName + "_TRANSPOSE_comp";
    } else {
        imageShaderName = imageShaderName + "_comp";
    }
    mSecond = bn->getPipeline(imageShaderName, secondTypes);
    mImageBufferSet.reset(mSecond->createSet());
}

void VulkanMatMul::Reorder::encode(VkBuffer source, size_t sourceSize, VkBuffer middleBuffer, size_t middelBufferSize, const VulkanImage* dest, const VulkanCommandPool::Buffer* cmdBuffer, const VulkanMatMul::Reorder::nchwBuffer& buffer) {
    // First: nchw to nc4hw4
    auto ptr = (nchwBuffer*)mUnitBuffer->map();
    ::memcpy(ptr, &buffer, sizeof(buffer));
    mUnitBuffer->unmap();
    auto c = buffer.size[1];
    auto b = buffer.size[0];
    auto w = buffer.size[3];
    auto h = buffer.size[2];
    auto cDiv4 = UP_DIV(c, 4);
    mBufferBufferSet->writeBuffer(middleBuffer, 1, middelBufferSize);
    mBufferBufferSet->writeBuffer(source, 0, sourceSize, 0);
    mBufferBufferSet->writeBuffer(mUnitBuffer->buffer(), 2, mUnitBuffer->size());
    auto totalNumber = cDiv4 * w * h * b;

    mFirst->bind(cmdBuffer->get(), mBufferBufferSet->get());
    cmdBuffer->barrierSource(source, 0, sourceSize);
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalNumber, 256), 1, 1);
    
    // Second: nc4hw4 to image2d
    cmdBuffer->barrierImageIfNeeded(dest, VK_IMAGE_LAYOUT_GENERAL);
    mImageBufferSet->writeImage(dest->view(), mBackend->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
    mImageBufferSet->writeBuffer(middleBuffer, 1, middelBufferSize);
    mImageBufferSet->writeBuffer(mUnitBuffer->buffer(), 2, mUnitBuffer->size());
    mSecond->bind(cmdBuffer->get(), mImageBufferSet->get());
    cmdBuffer->barrierSource(middleBuffer, 0, middelBufferSize);
    auto totalSchedule = cDiv4 * w * h * UP_DIV(b, 4);
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalSchedule, 256), 1, 1);

    cmdBuffer->barrierImageIfNeeded(dest, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}
int VulkanMatMul::Reorder::computeMiddleBufferSize(int b, int h, int w, int c) const {
    auto cDiv4 = UP_DIV(c, 4);
    auto totalNumber = cDiv4 * w * h * b;
    return totalNumber * 4;
}
void VulkanMatMul::Reorder::revert(VkBuffer dest, size_t destSize, VkBuffer middleBuffer, size_t middelBufferSize, const VulkanImage* source, const VulkanCommandPool::Buffer* cmdBuffer, const VulkanMatMul::Reorder::nchwBuffer& buffer) {
    // First: nchw to nc4hw4
    auto ptr = (nchwBuffer*)mUnitBuffer->map();
    ::memcpy(ptr, &buffer, sizeof(buffer));
    mUnitBuffer->unmap();
    auto c = buffer.size[1];
    auto b = buffer.size[0];
    auto w = buffer.size[3];
    auto h = buffer.size[2];
    auto cDiv4 = UP_DIV(c, 4);
    // First: image2d to nc4hw4
    mImageBufferSet->writeImage(source->view(), mBackend->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0);
    mImageBufferSet->writeBuffer(middleBuffer, 1, middelBufferSize);
    mImageBufferSet->writeBuffer(mUnitBuffer->buffer(), 2, mUnitBuffer->size());
    mSecond->bind(cmdBuffer->get(), mImageBufferSet->get());
    auto totalSchedule = cDiv4 * w * h * UP_DIV(b, 4);
    cmdBuffer->barrierImageIfNeeded(source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    // cmdBuffer->barrierImage(source->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalSchedule, 256), 1, 1);

    // Second: nc4hw4 to nchw
    mBufferBufferSet->writeBuffer(middleBuffer, 1, middelBufferSize);
    mBufferBufferSet->writeBuffer(dest, 0, destSize, 0);
    mBufferBufferSet->writeBuffer(mUnitBuffer->buffer(), 2, mUnitBuffer->size());
    auto totalNumber = cDiv4 * w * h * b;
    mFirst->bind(cmdBuffer->get(), mBufferBufferSet->get());
    cmdBuffer->barrierSource(middleBuffer, 0, middelBufferSize);
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalNumber, 256), 1, 1);
}

VulkanMatMul::VulkanMatMul(bool transposeA, bool transposeB, Backend* bn, bool hasBias) : VulkanBasicExecution(bn) {
    auto vkBn = (VulkanBackend*)bn;
    mTransposeA = transposeA;
    auto types = std::vector<VkDescriptorType>{
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    if (!mTransposeA) {
        mInputPipline = vkBn->getPipeline("glsl_matmul_input_comp", types);
    } else {
        mInputPipline = vkBn->getPipeline("glsl_matmul_input_TRANSPOSE_comp", types);
    }
    mTransposeB = transposeB;
    if (mTransposeB) {
        mWeightPipline = vkBn->getPipeline("glsl_matmul_kernel_comp", types);
    } else {
        mWeightPipline = vkBn->getPipeline("glsl_matmul_kernel_TRANSPOSE_comp", types);
    }
    if (hasBias) {
        auto ntypes = std::vector<VkDescriptorType>{
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        };

        mOutputPipeline = vkBn->getPipeline("glsl_matmul_output_BIAS_comp", ntypes);
    } else {
        mOutputPipeline = vkBn->getPipeline("glsl_matmul_output_comp", types);
    }
}
ErrorCode VulkanMatMul::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                           const VulkanCommandPool::Buffer *cmdBuffer) {
    mTempBuffer.clear();
    mSets.clear();
    auto input0T = reinterpret_cast<VulkanTensor*>(inputs[0]->deviceId());
    auto input1T = reinterpret_cast<VulkanTensor*>(inputs[1]->deviceId());
    auto outputT = reinterpret_cast<VulkanTensor*>(outputs[0]->deviceId());
    if (input0T->imageSize() > 1 || input1T->imageSize() > 1 || outputT->imageSize() > 1) {
        return NOT_SUPPORT;
    }
    Tensor* C       = outputs[0];
    auto w0         = inputs[0]->length(1);
    auto h0         = inputs[0]->length(0);
    auto e = C->length(0);
    auto h = C->length(1);
    auto l = w0;
    auto vkBn = static_cast<VulkanBackend*>(backend());
    if (mTransposeA) {
        l = h0;
    }
    struct OffsetBuffer {
        ivec4 offset;
        ivec4 size;
    };
    {
        // Pretreat B
        mKernelImage.reset(new VulkanImage(vkBn->getDynamicMemoryPool(), false, {ALIGN_UP4(l), UP_DIV(h, 4)}));
        std::shared_ptr<VulkanPipeline::DescriptorSet> des(mWeightPipline->createSet());
        OffsetBuffer buffer;
        buffer.size[0] = UP_DIV(l, 4);
        buffer.size[1] = UP_DIV(h, 4);
        buffer.size[3] = UP_DIV(l, 4) * UP_DIV(h, 4);
        std::shared_ptr<VulkanBuffer> uniformBuffer(new VulkanBuffer(vkBn->getMemoryPool(), false, sizeof(OffsetBuffer), &buffer, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        des->writeBuffer(uniformBuffer->buffer(), 2, uniformBuffer->size());
        des->writeImage(mKernelImage->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
        des->writeImage(input1T->image()->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        mTempBuffer.emplace_back(uniformBuffer);
        mSets.emplace_back(des);
        mWeightPipline->bind(cmdBuffer->get(), des->get());
        cmdBuffer->barrierImage(input1T->image()->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(buffer.size[3], 256), 1, 1);
    }
    {
        // Pretreat A
        mInputImage.reset(new VulkanImage(vkBn->getDynamicMemoryPool(), false, {ALIGN_UP4(l), UP_DIV(e, 4)}));
        std::shared_ptr<VulkanPipeline::DescriptorSet> des(mInputPipline->createSet());
        OffsetBuffer buffer;
        buffer.size[0] = UP_DIV(l, 4);
        buffer.size[1] = UP_DIV(e, 4);
        buffer.size[3] = UP_DIV(l, 4) * UP_DIV(e, 4);
        std::shared_ptr<VulkanBuffer> uniformBuffer(new VulkanBuffer(vkBn->getMemoryPool(), false, sizeof(OffsetBuffer), &buffer, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        des->writeBuffer(uniformBuffer->buffer(), 2, uniformBuffer->size());
        des->writeImage(mInputImage->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
        des->writeImage(input0T->image()->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        mTempBuffer.emplace_back(uniformBuffer);
        mSets.emplace_back(des);
        mInputPipline->bind(cmdBuffer->get(), des->get());
        cmdBuffer->barrierImage(input0T->image()->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(buffer.size[3], 256), 1, 1);
    }
    mCore.reset(new VulkanMatrixMultier4x4(vkBn, nullptr, l, h, 1, mKernelImage));
    mOutputImage.reset(new VulkanImage(vkBn->getDynamicMemoryPool(), false, {ALIGN_UP4(h), UP_DIV(e, 4)}));
    mCore->prepare(cmdBuffer, e, mOutputImage, mInputImage);
    mCore->compute(cmdBuffer);
    mInputImage->release();
    mKernelImage->release();
    {
        // Posttreat C
        std::shared_ptr<VulkanPipeline::DescriptorSet> des(mOutputPipeline->createSet());
        OffsetBuffer buffer;
        buffer.size[0] = UP_DIV(h, 4);
        buffer.size[1] = UP_DIV(e, 4);
        buffer.size[3] = UP_DIV(h, 4) * UP_DIV(e, 4);
        std::shared_ptr<VulkanBuffer> uniformBuffer(new VulkanBuffer(vkBn->getMemoryPool(), false, sizeof(OffsetBuffer), &buffer, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        des->writeBuffer(uniformBuffer->buffer(), 2, uniformBuffer->size());
        des->writeImage(outputT->image()->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
        des->writeImage(mOutputImage->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        if (inputs.size() > 2) {
            des->writeImage(reinterpret_cast<VulkanTensor*>(inputs[2]->deviceId())->image()->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 3);
        }
        mSets.emplace_back(des);
        mOutputPipeline->bind(cmdBuffer->get(), des->get());
        cmdBuffer->barrierImage(mOutputImage->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(buffer.size[3], 256), 1, 1);

        mTempBuffer.emplace_back(uniformBuffer);
    }
    mOutputImage->release();

    return NO_ERROR;
}
class VulkanMatMulCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        Tensor* C       = outputs[0];
        auto w0         = inputs[0]->length(1);
        auto h0         = inputs[0]->length(0);
        auto e = C->length(0);
        auto h = C->length(1);
        auto l = w0;
        auto vkBn = static_cast<VulkanBackend*>(bn);
        const auto mat = op->main_as_MatMul();
        if (mat->transposeA()) {
            l = h0;
        }
        auto limit = vkBn->proty().limits.maxImageDimension3D;
        if (e > limit || h > limit || l > limit) {
            return nullptr;
        }

        return new VulkanMatMul(mat->transposeA(), mat->transposeB(), bn, inputs.size() > 2);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_MatMul, new VulkanMatMulCreator);
    return true;
}();

}
