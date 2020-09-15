//
//  VulkanBinary.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/execution/VulkanBinary.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {

struct ConstBuffer {
    ivec4 stride00;
    ivec4 stride01;
    ivec4 stride10;
    ivec4 stride11;
    ivec4 stride20;
    ivec4 stride21;
};
static std::string _getShaderName(const Op* op, bool image) {
    std::string prefix = "glsl_binaryBroadcast_";
    if (image) {
        prefix = "glsl_binaryImage_";
    }
    std::string posfix = "_comp";
    std::string mid = "";
    if (op->type() == OpType_Eltwise) {
        if (op->main_as_Eltwise()->coeff() != nullptr) {
            // Don't support
            return "";
        }
        switch (op->main_as_Eltwise()->type()) {
            case EltwiseType_SUB:
                mid = "SUB";
                break;
            case EltwiseType_MAXIMUM:
                mid = "VMAX";
                break;
            case EltwiseType_PROD:
                mid = "MUL";
                break;
            case EltwiseType_SUM:
                mid = "ADD";
                break;
            default:
                break;
        }
    } else if (op->type() == OpType_BinaryOp) {
        switch (op->main_as_BinaryOp()->opType()) {
            case BinaryOpOperation_ADD:
                mid = "ADD";
                break;
            case BinaryOpOperation_SUB:
                mid = "SUB";
                break;
            case BinaryOpOperation_MAXIMUM:
                mid = "VMAX";
                break;
            case BinaryOpOperation_MINIMUM:
                mid = "VMIN";
                break;
            case BinaryOpOperation_MUL:
                mid = "MUL";
                break;
            case BinaryOpOperation_DIV:
            case BinaryOpOperation_REALDIV:
                mid = "DIV";
                break;
            default:
                break;
        }
    }
    if (mid.empty()) {
        return mid;
    }
    return prefix + mid + posfix;
}

VulkanBinary::VulkanBinary(const std::string& shaderName, Backend* bn, bool image) : VulkanBasicExecution(bn) {
    auto vkBn   = static_cast<VulkanBackend*>(bn);
    mImage = image;
    mConstBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, sizeof(ConstBuffer), nullptr,
                                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    if (image) {
        mBinaryPipeline = vkBn->getPipeline(shaderName, {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        });
    } else {
        mBinaryPipeline = vkBn->getPipeline(shaderName, {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        });
    }
    mDescriptorSet.reset(mBinaryPipeline->createSet());
}

VulkanBinary::~VulkanBinary() {
}

ErrorCode VulkanBinary::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    MNN_ASSERT(1 == outputs.size());

    auto vkBn = (VulkanBackend*)backend();
    const int outputElements = outputs[0]->elementSize();
    {
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        auto output = outputs[0];
        MNN_ASSERT(input0->getType().code == halide_type_float);
        if (!mImage) {
            // for buffer input
            #define MAX_DIM 6
            int dims[MAX_DIM];
            int stride[MAX_DIM];
            int iStride0[MAX_DIM];
            int iStride1[MAX_DIM];
            OpCommonUtils::broastCastComputeDim(dims, stride, iStride0, iStride1, input0, input1, output);

            auto binaryOpParam = reinterpret_cast<ConstBuffer*>(mConstBuffer->map());
            binaryOpParam->stride01[3] = outputElements;
            binaryOpParam->stride01[2] = 1;
            binaryOpParam->stride01[1] = 1;
            binaryOpParam->stride01[0] = dims[5];
            binaryOpParam->stride00[3] = dims[4] * binaryOpParam->stride01[0];
            binaryOpParam->stride00[2] = dims[3] * binaryOpParam->stride00[3];
            binaryOpParam->stride00[1] = dims[2] * binaryOpParam->stride00[2];
            binaryOpParam->stride00[0] = dims[1] * binaryOpParam->stride00[1];

            ::memcpy(binaryOpParam->stride10, iStride0, 4 * sizeof(int));
            ::memcpy(binaryOpParam->stride11, iStride0 + 4, 2 * sizeof(int));
            ::memcpy(binaryOpParam->stride20, iStride1, 4 * sizeof(int));
            ::memcpy(binaryOpParam->stride21, iStride1 + 4, 2 * sizeof(int));
            mConstBuffer->unmap();

            mDescriptorSet->writeBuffer(reinterpret_cast<VkBuffer>(output->deviceId()), 0, output->size());
            mDescriptorSet->writeBuffer(reinterpret_cast<VkBuffer>(input0->deviceId()), 1, input0->size());
            mDescriptorSet->writeBuffer(reinterpret_cast<VkBuffer>(input1->deviceId()), 2, input1->size());
            mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 3, mConstBuffer->size());
            mBinaryPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
            cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(input0->deviceId()), 0, input0->size());
            cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(input1->deviceId()), 0, input1->size());
            vkCmdDispatch(cmdBuffer->get(), UP_DIV(outputElements, 256), 1, 1);
        } else {
            // for NC4HW4 input
            const int iw = input0->width();
            const int ih = input0->height();

            MNN_ASSERT(input0->dimensions() == input1->dimensions());

            const int icDiv4 = UP_DIV(input0->channel(), 4);
            auto total = icDiv4 * input0->batch() * iw * ih;

            auto binaryOpParam = reinterpret_cast<ConstBuffer*>(mConstBuffer->map());
            ::memset(binaryOpParam, 0, sizeof(ConstBuffer));
            binaryOpParam->stride00[3] = total;
            binaryOpParam->stride00[0] = iw;
            binaryOpParam->stride00[1] = ih;
            binaryOpParam->stride00[2] = icDiv4;
            mConstBuffer->unmap();

            auto sampler = vkBn->getCommonSampler();
            mDescriptorSet->writeImage(reinterpret_cast<VkImageView>(output->deviceId()), sampler->get(),
                                       VK_IMAGE_LAYOUT_GENERAL, 0);
            auto outputT = vkBn->findTensor(output->deviceId());
            auto input0T = vkBn->findTensor(input0->deviceId());
            auto input1T = vkBn->findTensor(input1->deviceId());
            cmdBuffer->barrierImageIfNeeded(outputT->image(), VK_IMAGE_LAYOUT_GENERAL);
            cmdBuffer->barrierImageIfNeeded(input0T->image(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            cmdBuffer->barrierImageIfNeeded(input1T->image(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            // cmdBuffer->barrierImage(input0T->image()->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            // cmdBuffer->barrierImage(input1T->image()->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            mDescriptorSet->writeImage(reinterpret_cast<VkImageView>(input0->deviceId()), sampler->get(),
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
            mDescriptorSet->writeImage(reinterpret_cast<VkImageView>(input1->deviceId()), sampler->get(),
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
            mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 3, mConstBuffer->size());
            mBinaryPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
            vkCmdDispatch(cmdBuffer->get(), UP_DIV(total, 256), 1, 1);
        }
    }
    if (inputs.size() > 2) {
        mExtraDescriptorSet.clear();
        for (int i=2; i<inputs.size(); ++i) {
            auto input0 = outputs[0];
            auto input1 = inputs[i];
            auto output = outputs[0];
            std::shared_ptr<VulkanPipeline::DescriptorSet> newSet(mBinaryPipeline->createSet());
            mExtraDescriptorSet.push_back(newSet);

            if (!mImage) {
                newSet->writeBuffer(reinterpret_cast<VkBuffer>(output->deviceId()), 0, output->size());
                newSet->writeBuffer(reinterpret_cast<VkBuffer>(input0->deviceId()), 1, input0->size());
                newSet->writeBuffer(reinterpret_cast<VkBuffer>(input1->deviceId()), 2, input1->size());
                newSet->writeBuffer(mConstBuffer->buffer(), 3, mConstBuffer->size());
                mBinaryPipeline->bind(cmdBuffer->get(), newSet->get());
                cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(input0->deviceId()), 0, input0->size());
                cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(input1->deviceId()), 0, input1->size());
                vkCmdDispatch(cmdBuffer->get(), UP_DIV(outputElements, 256), 1, 1);
            } else {
                // for NC4HW4 input
                const int iw = input0->width();
                const int ih = input0->height();
                const int icDiv4 = UP_DIV(input0->channel(), 4);
                auto total = icDiv4 * input0->batch() * iw * ih;
                auto sampler = vkBn->getCommonSampler();
                auto outputT = vkBn->findTensor(output->deviceId());
                auto input0T = vkBn->findTensor(input0->deviceId());
                auto input1T = vkBn->findTensor(input1->deviceId());
                cmdBuffer->barrierImageIfNeeded(outputT->image(), VK_IMAGE_LAYOUT_GENERAL);
                cmdBuffer->barrierImageIfNeeded(input0T->image(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                cmdBuffer->barrierImageIfNeeded(input1T->image(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                // cmdBuffer->barrierImage(input0T->image()->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                // cmdBuffer->barrierImage(input1T->image()->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                newSet->writeImage(reinterpret_cast<VkImageView>(output->deviceId()), sampler->get(),
                                           VK_IMAGE_LAYOUT_GENERAL, 0);
                newSet->writeImage(reinterpret_cast<VkImageView>(input0->deviceId()), sampler->get(),
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
                newSet->writeImage(reinterpret_cast<VkImageView>(input1->deviceId()), sampler->get(),
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
                newSet->writeBuffer(mConstBuffer->buffer(), 3, mConstBuffer->size());
                mBinaryPipeline->bind(cmdBuffer->get(), newSet->get());
                vkCmdDispatch(cmdBuffer->get(), UP_DIV(total, 256), 1, 1);
            }
        }
    }
    return NO_ERROR;
}

class VulkanBinaryCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        auto input0 = inputs[0];
        if (input0->getType().code != halide_type_float) {
            return nullptr;
        }
        auto image = TensorUtils::getDescribe(input0)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
        auto shader = _getShaderName(op, image);
        if (shader.empty()) {
            return nullptr;
        }
        return new VulkanBinary(shader, backend, image);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_BinaryOp, new VulkanBinaryCreator);
    VulkanBackend::addCreator(OpType_Eltwise, new VulkanBinaryCreator);
    return true;
}();

} // namespace MNN
