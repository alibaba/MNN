//
//  VulkanMatMul.cpp
//  MNN
//
//  Created by MNN on 2020/03/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanMatMul.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {

class TexturePosComputer {
public:
    TexturePosComputer(const Tensor* tensor) {
        mFormat = TensorUtils::getDescribe(tensor)->dimensionFormat;
        mNHWC = VulkanTensor::tensorShapeFormat(tensor);
    }
    ~ TexturePosComputer() {
        // Do nothing
    }

    // X, Y, rgba
    std::array<int, 3> computePos(int offset) const {
        std::array<int, 4> nhwcPos;
        if (mFormat == MNN_DATA_FORMAT_NHWC) {
            // NHWC
            nhwcPos[3] = offset % mNHWC[3];
            offset = (offset - nhwcPos[3]) / mNHWC[3];
            nhwcPos[2] = offset % mNHWC[2];
            offset = (offset - nhwcPos[2]) / mNHWC[2];
            nhwcPos[1] = offset % mNHWC[1];
            nhwcPos[0] = offset / mNHWC[1];
        } else {
            // NCHW
            nhwcPos[2] = offset % mNHWC[2];
            offset = offset / mNHWC[2];
            nhwcPos[1] = offset % mNHWC[1];
            offset = offset / mNHWC[1];
            nhwcPos[3] = offset % mNHWC[3];
            nhwcPos[0] = offset / mNHWC[3];
        }
//        MNN_PRINT("n, c, h, w: %d, %d, %d, %d\n", nhwcPos[0], nhwcPos[3], nhwcPos[1], nhwcPos[2]);
        std::array<int, 3> res;
        res[2] = nhwcPos[3] % 4;
        auto c4 = nhwcPos[3] / 4;
        res[0] = c4 * mNHWC[2] + nhwcPos[2];
        res[1] = nhwcPos[0] * mNHWC[1] + nhwcPos[1];
        return res;
    }
private:
    MNN_DATA_FORMAT mFormat;
    std::array<int, 4> mNHWC;
};

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
    dest->barrierWrite(cmdBuffer->get());
    mImageBufferSet->writeImage(dest->view(), mBackend->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
    mImageBufferSet->writeBuffer(middleBuffer, 1, middelBufferSize);
    mImageBufferSet->writeBuffer(mUnitBuffer->buffer(), 2, mUnitBuffer->size());
    mSecond->bind(cmdBuffer->get(), mImageBufferSet->get());
    cmdBuffer->barrierSource(middleBuffer, 0, middelBufferSize);
    auto totalSchedule = cDiv4 * w * h * UP_DIV(b, 4);
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalSchedule, 256), 1, 1);
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
    source->barrierRead(cmdBuffer->get());
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

VulkanMatMul::VulkanMatMul(bool transposeA, bool transposeB, Backend* bn) : VulkanBasicExecution(bn) {
    mTransposeA = transposeA;
    mTransposeB = transposeB;
}

bool VulkanMatMul::encode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
            const VulkanCommandPool::Buffer *cmdBuffer, const MatMulInfo& info) {
    mTempBuffer.clear();
    mSets.clear();
    auto input0T = reinterpret_cast<VulkanTensor*>(inputs[0]->deviceId());
    auto input1T = reinterpret_cast<VulkanTensor*>(inputs[1]->deviceId());
    auto outputT = reinterpret_cast<VulkanTensor*>(outputs[0]->deviceId());
    if (input0T->imageSize() > 1 || input1T->imageSize() > 1 || outputT->imageSize() > 1) {
        // TODO: Copy to temp buffer
        return false;
    }
    auto types = std::vector<VkDescriptorType>{
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    const VulkanPipeline* mInputPipline = nullptr;
    const VulkanPipeline* mWeightPipline = nullptr;
    const VulkanPipeline* mOutputPipeline = nullptr;
    auto vkBn = static_cast<VulkanBackend*>(backend());
    int e = info.e;
    int l = info.l;
    int h = info.h;
    struct OffsetBuffer {
        ivec4 size;
        float transpose[16];
    };
    {
        // Pretreat B
        // Turn bStride(e, l, h) -> x,y + offset
//        MNN_PRINT("Compute B %d x %d:\n", inputs[1]->length(0), inputs[1]->length(1));
        TexturePosComputer comp(inputs[1]);
        auto offset0 = comp.computePos(info.offsetB);
        if (offset0[2] != 0) {
            return false;
        }
        auto offsetl = comp.computePos(info.bStride[1]);
        auto offseth = comp.computePos(info.bStride[2]);
        bool transposeB = offsetl[2] == 1;
        if (l == 1) {
            // TODO: Find better way
            transposeB = offseth[2] == 0;
        }
        int l_W = offsetl[0] * 4 + offsetl[2];
        int l_H = offsetl[1] * 4;
        int h_W = offseth[0] * 4 + offseth[2];
        int h_H = offseth[1] * 4;
//        MNN_PRINT("Compute B: %d x %d, %d-%d-%d,  %d-%d-%d\n", info.bStride[1], info.bStride[2], offsetl[0], offsetl[1], offsetl[2], offseth[0], offseth[1], offseth[2]);
        if (transposeB) {
            mWeightPipline = vkBn->getPipeline("glsl_matmul_input_TRANSPOSE_comp", types);
        } else {
            mWeightPipline = vkBn->getPipeline("glsl_matmul_input_comp", types);
        }
        mKernelImage.reset(new VulkanImage(vkBn->getDynamicMemoryPool(), false, {ALIGN_UP4(l), UP_DIV(h, 4)}));
        std::shared_ptr<VulkanLayout::DescriptorSet> des(mWeightPipline->createSet());
        OffsetBuffer buffer;
        buffer.size[0] = UP_DIV(l, 4);
        buffer.size[1] = UP_DIV(h, 4);
        if (transposeB) {
            buffer.size[2] = h;
        } else {
            buffer.size[2] = l;
        }
        buffer.size[2] = buffer.size[2] + offset0[1];
        buffer.size[3] = UP_DIV(l, 4) * UP_DIV(h, 4);
        ::memset(buffer.transpose, 0, 16 * sizeof(float));
        buffer.transpose[4 * 0 + 0] = l_W;
        buffer.transpose[4 * 0 + 1] = h_W;
        buffer.transpose[4 * 0 + 2] = 0;
        buffer.transpose[4 * 0 + 3] = offset0[0];

        buffer.transpose[4 * 1 + 0] = l_H;
        buffer.transpose[4 * 1 + 1] = h_H;
        buffer.transpose[4 * 1 + 2] = 0;
        buffer.transpose[4 * 1 + 3] = offset0[1];
        std::shared_ptr<VulkanBuffer> uniformBuffer(new VulkanBuffer(vkBn->getMemoryPool(), false, sizeof(OffsetBuffer), &buffer, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        des->writeBuffer(uniformBuffer->buffer(), 2, uniformBuffer->size());
        des->writeImage(mKernelImage->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
        des->writeImage(input1T->image()->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        mTempBuffer.emplace_back(uniformBuffer);
        mSets.emplace_back(des);
        mWeightPipline->bind(cmdBuffer->get(), des->get());

        input1T->image()->barrierRead(cmdBuffer->get());
        mKernelImage->barrierWrite(cmdBuffer->get());

        vkCmdDispatch(cmdBuffer->get(), UP_DIV(buffer.size[3], 256), 1, 1);
    }
    {
        // Pretreat A
        TexturePosComputer comp(inputs[0]);
        auto offset0 = comp.computePos(info.offsetA);
        if (offset0[2] != 0) {
            return false;
        }
        auto offsetl = comp.computePos(info.aStride[1]);
        auto offsete = comp.computePos(info.aStride[0]);
        int l_W = offsetl[0] * 4 + offsetl[2];
        int l_H = offsetl[1] * 4;
        int e_W = offsete[0] * 4 + offsete[2];
        int e_H = offsete[1] * 4;
        bool transposeA = offsetl[2] != 1;
        if (l == 1) {
            // TODO: Find better way
            transposeA = offsete[2] == 1;
        }
        if (!transposeA) {
            mInputPipline = vkBn->getPipeline("glsl_matmul_input_comp", types);
        } else {
            mInputPipline = vkBn->getPipeline("glsl_matmul_input_TRANSPOSE_comp", types);
        }
        mInputImage.reset(new VulkanImage(vkBn->getDynamicMemoryPool(), false, {ALIGN_UP4(l), UP_DIV(e, 4)}));
        std::shared_ptr<VulkanLayout::DescriptorSet> des(mInputPipline->createSet());
        OffsetBuffer buffer;
        buffer.size[0] = UP_DIV(l, 4);
        buffer.size[1] = UP_DIV(e, 4);
        if (transposeA) {
            buffer.size[2] = l;
        } else {
            buffer.size[2] = e;
        }
        buffer.size[2] = buffer.size[2] + offset0[1];
        buffer.size[3] = UP_DIV(l, 4) * UP_DIV(e, 4);
        ::memset(buffer.transpose, 0, 16 * sizeof(float));
        buffer.transpose[4 * 0 + 0] = l_W;
        buffer.transpose[4 * 0 + 1] = e_W;
        buffer.transpose[4 * 0 + 2] = 0;
        buffer.transpose[4 * 0 + 3] = offset0[0];

        buffer.transpose[4 * 1 + 0] = l_H;
        buffer.transpose[4 * 1 + 1] = e_H;
        buffer.transpose[4 * 1 + 2] = 0;
        buffer.transpose[4 * 1 + 3] = offset0[1];
        std::shared_ptr<VulkanBuffer> uniformBuffer(new VulkanBuffer(vkBn->getMemoryPool(), false, sizeof(OffsetBuffer), &buffer, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        des->writeBuffer(uniformBuffer->buffer(), 2, uniformBuffer->size());
        des->writeImage(mInputImage->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
        des->writeImage(input0T->image()->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        mTempBuffer.emplace_back(uniformBuffer);
        mSets.emplace_back(des);
        mInputPipline->bind(cmdBuffer->get(), des->get());

        input0T->image()->barrierRead(cmdBuffer->get());
        mInputImage->barrierWrite(cmdBuffer->get());

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
        if (inputs.size() > 2) {
            auto ntypes = std::vector<VkDescriptorType>{
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            };
            mOutputPipeline = vkBn->getPipeline("glsl_matmul_output_BIAS_comp", ntypes);
        } else {
            mOutputPipeline = vkBn->getPipeline("glsl_matmul_output_comp", types);
        }
        TexturePosComputer comp(outputs[0]);
        auto offset0 = comp.computePos(info.offsetC);
        if (offset0[2] != 0) {
            return false;
        }
        auto offsete = comp.computePos(info.cStride[0]);
        auto offseth = comp.computePos(info.cStride[2]);

        std::shared_ptr<VulkanLayout::DescriptorSet> des(mOutputPipeline->createSet());
        OffsetBuffer buffer;
        buffer.size[0] = UP_DIV(h, 4);
        buffer.size[1] = UP_DIV(e, 4);
        buffer.size[2] = e + offset0[1];
        buffer.size[3] = UP_DIV(h, 4) * UP_DIV(e, 4);
        ::memset(buffer.transpose, 0, 16 * sizeof(float));
        buffer.transpose[4 * 0 + 0] = 1;
        buffer.transpose[4 * 0 + 1] = 0;
        buffer.transpose[4 * 0 + 2] = 0;
        buffer.transpose[4 * 0 + 3] = offset0[0];

        buffer.transpose[4 * 1 + 0] = 0;
        buffer.transpose[4 * 1 + 1] = 4;
        buffer.transpose[4 * 1 + 2] = 0;
        buffer.transpose[4 * 1 + 3] = offset0[1];

        std::shared_ptr<VulkanBuffer> uniformBuffer(new VulkanBuffer(vkBn->getMemoryPool(), false, sizeof(OffsetBuffer), &buffer, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        des->writeBuffer(uniformBuffer->buffer(), 2, uniformBuffer->size());
        des->writeImage(outputT->image()->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
        des->writeImage(mOutputImage->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        if (inputs.size() > 2) {
            float biasTranspose[16];
            std::shared_ptr<VulkanBuffer> biasuniformBuffer(new VulkanBuffer(vkBn->getMemoryPool(), false, sizeof(biasTranspose), &biasTranspose, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
            des->writeImage(reinterpret_cast<VulkanTensor*>(inputs[2]->deviceId())->image()->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 3);
            des->writeBuffer(biasuniformBuffer->buffer(), 4, 16 * sizeof(float));
            mTempBuffer.emplace_back(biasuniformBuffer);
        }
        mSets.emplace_back(des);
        mOutputPipeline->bind(cmdBuffer->get(), des->get());

        outputT->image()->barrierWrite(cmdBuffer->get());
        mOutputImage->barrierRead(cmdBuffer->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(buffer.size[3], 256), 1, 1);

        mTempBuffer.emplace_back(uniformBuffer);
    }
    mOutputImage->release();
    return true;
}

ErrorCode VulkanMatMul::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                           const VulkanCommandPool::Buffer *cmdBuffer) {
    Tensor* C       = outputs[0];
    auto w0         = inputs[0]->length(1);
    auto h0         = inputs[0]->length(0);
    auto e = C->length(0);
    auto h = C->length(1);
    auto l = w0;
    if (mTransposeA) {
        l = h0;
    }
    MatMulInfo info;
    info.e = e;
    info.l = l;
    info.h = h;
    if (mTransposeA) {
        info.aStride[0] = 1;
        info.aStride[1] = e;
        info.aStride[2] = 0;
    } else {
        info.aStride[0] = l;
        info.aStride[1] = 1;
        info.aStride[2] = 0;
    }
    if (mTransposeB) {
        info.bStride[0] = 0;
        info.bStride[1] = 1;
        info.bStride[2] = l;
    } else {
        info.bStride[0] = 0;
        info.bStride[1] = h;
        info.bStride[2] = 1;
    }
    info.cStride[0] = h;
    info.cStride[1] = 0;
    info.cStride[2] = 1;
    auto res = encode(inputs, outputs, cmdBuffer, info);
    if (!res) {
        return NOT_SUPPORT;
    }
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
        auto limit = vkBn->proty().limits.maxImageDimension2D;
        if (e > limit || h > limit || l > limit) {
            return nullptr;
        }
        return new VulkanMatMul(mat->transposeA(), mat->transposeB(), bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_MatMul, new VulkanMatMulCreator);
    return true;
}();

}
