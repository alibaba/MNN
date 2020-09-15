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
    mUnitBuffer.reset(new VulkanBuffer(bn->getMemoryPool(), true, sizeof(nchwBuffer), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
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
    cmdBuffer->barrierSource(dest, 0, destSize);
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalNumber, 256), 1, 1);
}

VulkanMatMul::VulkanMatMul(bool transposeA, bool transposeB, Backend* bn) : VulkanBasicExecution(bn) {
    mTransposeA = transposeA;
    mTransposeB = transposeB;
    auto vkBn = (VulkanBackend*)bn;
    mInputReorder.reset(new Reorder(vkBn, false, false));
    mWeightReorder.reset(new Reorder(vkBn, true, false));
    mOutputReorder.reset(new Reorder(vkBn, false, true));
}
ErrorCode VulkanMatMul::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                           const VulkanCommandPool::Buffer *cmdBuffer) {
    mTempBuffer.clear();
    Tensor* C       = outputs[0];
    auto w0         = inputs[0]->length(1);
    auto h0         = inputs[0]->length(0);
    auto e = C->length(0);
    auto h = C->length(1);
    auto l = w0;
    auto vkBn = (VulkanBackend*)backend();
    if (mTransposeA) {
        l = h0;
    }
    {
        // Pretreat B
        mKernelImage.reset(new VulkanImage(vkBn->getDynamicMemoryPool(), false, {ALIGN_UP4(l), UP_DIV(h, 4)}));
        std::shared_ptr<VulkanBuffer> mid(new VulkanBuffer(vkBn->getDynamicMemoryPool(), false, mInputReorder->computeMiddleBufferSize(h, 1, 1, l)*sizeof(float), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        mid->release();
        Reorder::nchwBuffer nchw;
        nchw.size[0] = h;
        nchw.size[1] = l;
        nchw.size[2] = 1;
        nchw.size[3] = 1;
        if (mTransposeB) {
            nchw.stride[0] = l;
            nchw.stride[1] = 1;
        } else {
            nchw.stride[0] = 1;
            nchw.stride[1] = h;
        }
        nchw.stride[2] = 1;
        nchw.stride[3] = 1;
        mWeightReorder->encode((VkBuffer)inputs[1]->deviceId(), inputs[1]->size(), mid->buffer(), mid->size(), mKernelImage.get(), cmdBuffer, nchw);
        mTempBuffer.emplace_back(mid);
    }
    {
        // Pretreat A
        mInputImage.reset(new VulkanImage(vkBn->getDynamicMemoryPool(), false, {ALIGN_UP4(l), UP_DIV(e, 4)}));
        std::shared_ptr<VulkanBuffer> mid(new VulkanBuffer(vkBn->getDynamicMemoryPool(), false, mInputReorder->computeMiddleBufferSize(e, 1, 1, l)*sizeof(float), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        mid->release();
        Reorder::nchwBuffer nchw;
        nchw.size[0] = e;
        nchw.size[1] = l;
        nchw.size[2] = 1;
        nchw.size[3] = 1;
        if (mTransposeA) {
            nchw.stride[0] = 1;
            nchw.stride[1] = e;
        } else {
            nchw.stride[0] = l;
            nchw.stride[1] = 1;
        }
        nchw.stride[2] = 1;
        nchw.stride[3] = 1;
        mInputReorder->encode((VkBuffer)inputs[0]->deviceId(), inputs[0]->size(), mid->buffer(), mid->size(), mInputImage.get(), cmdBuffer, nchw);
        mTempBuffer.emplace_back(mid);
    }
    mCore.reset(new VulkanMatrixMultier4x4(vkBn, nullptr, l, h, 1, mKernelImage));
    mOutputImage.reset(new VulkanImage(vkBn->getDynamicMemoryPool(), false, {ALIGN_UP4(h), UP_DIV(e, 4)}));
    mCore->prepare(cmdBuffer, e, mOutputImage, mInputImage);
    mCore->compute(cmdBuffer);
    mInputImage->release();
    mKernelImage->release();
    {
        // Posttreat C
        std::shared_ptr<VulkanBuffer> mid(new VulkanBuffer(vkBn->getDynamicMemoryPool(), false, mInputReorder->computeMiddleBufferSize(e, 1, 1, h)*sizeof(float), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        mid->release();
        Reorder::nchwBuffer nchw;
        nchw.size[0] = e;
        nchw.size[1] = h;
        nchw.size[2] = 1;
        nchw.size[3] = 1;
        nchw.stride[0] = h;
        nchw.stride[1] = 1;
        nchw.stride[2] = 1;
        nchw.stride[3] = 1;
        mOutputReorder->revert((VkBuffer)(outputs[0]->deviceId()), outputs[0]->size(), mid->buffer(), mid->size(), mOutputImage.get(), cmdBuffer, nchw);
        mTempBuffer.emplace_back(mid);
    }
    mOutputImage->release();

    return NO_ERROR;
}
class VulkanMatMulCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        const auto mat = op->main_as_MatMul();
        return new VulkanMatMul(mat->transposeA(), mat->transposeB(), bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_MatMul, new VulkanMatMulCreator);
    return true;
}();

}
