//
//  VulkanBinary.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanBinary.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {

struct ConstBuffer {
    ivec4 imgSize; // for image data
    ivec4 stride;  // input0, input1, output, len, // for buffer data
};

VulkanBinary::VulkanBinary(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    mType        = op->main_as_BinaryOp()->opType();
    mVkBackend   = static_cast<VulkanBackend*>(bn);
    mConstBuffer = std::make_shared<VulkanBuffer>(mVkBackend->getMemoryPool(), false, sizeof(ConstBuffer), nullptr,
                                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
}

VulkanBinary::~VulkanBinary() {
}

ErrorCode VulkanBinary::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    MNN_ASSERT(2 == inputs.size());
    MNN_ASSERT(1 == outputs.size());

    auto input0 = inputs[0];
    auto input1 = inputs[1];
    auto output = outputs[0];
    MNN_ASSERT(input0->getType().code == halide_type_float);
    const auto intputFormat = TensorUtils::getDescribe(input0)->dimensionFormat;
    if (intputFormat == MNN_DATA_FORMAT_NHWC) {
        // for NHWC input
        std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};

        switch (mType) {
            case BinaryOpOperation_MUL:
                mBinaryPipeline = mVkBackend->getPipeline("glsl_elementwiseMulBuffer_comp", types);
                break;
            case BinaryOpOperation_ADD:
                mBinaryPipeline = mVkBackend->getPipeline("glsl_elementwiseAddBuffer_comp", types);
                break;
            case BinaryOpOperation_SUB:
                mBinaryPipeline = mVkBackend->getPipeline("glsl_elementwiseSubBuffer_comp", types);
                break;
            default:
                MNN_PRINT("Not Supported Binary Operation: %d\n", mType);
                MNN_ASSERT(false);
                break;
        }

        const int input0Elements = input0->elementSize();
        const int input1Elements = input1->elementSize();
        const int outputElements = output->elementSize();

        auto binaryOpParam = reinterpret_cast<ConstBuffer*>(mConstBuffer->map());
        ::memset(binaryOpParam, 0, sizeof(ConstBuffer));

        if (input0Elements == 1) {
            binaryOpParam->stride[0] = 0;
            binaryOpParam->stride[1] = 1;
            binaryOpParam->stride[2] = 1;
            binaryOpParam->stride[3] = outputElements;
        } else if (input1Elements == 1) {
            binaryOpParam->stride[0] = 1;
            binaryOpParam->stride[1] = 0;
            binaryOpParam->stride[2] = 1;
            binaryOpParam->stride[3] = outputElements;
        } else if (input0Elements == input1Elements) {
            binaryOpParam->stride[0] = 1;
            binaryOpParam->stride[1] = 1;
            binaryOpParam->stride[2] = 1;
            binaryOpParam->stride[3] = outputElements;
        } else {
            return NOT_SUPPORT;
        }
        mConstBuffer->flush(true, 0, sizeof(ConstBuffer));
        mConstBuffer->unmap();

        mDescriptorSet.reset(mBinaryPipeline->createSet());
        mDescriptorSet->writeBuffer(reinterpret_cast<VkBuffer>(output->deviceId()), 0, output->size());
        mDescriptorSet->writeBuffer(reinterpret_cast<VkBuffer>(input0->deviceId()), 1, input0->size());
        mDescriptorSet->writeBuffer(reinterpret_cast<VkBuffer>(input1->deviceId()), 2, input1->size());
        mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 3, mConstBuffer->size());
        mBinaryPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
        cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(input0->deviceId()), 0, input0->size());
        cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(input1->deviceId()), 0, input1->size());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(output->elementSize(), 8), 1, 1);
    } else {
        // for NC4HW4 input
        std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};

        switch (mType) {
            case BinaryOpOperation_ADD:
                mBinaryPipeline = mVkBackend->getPipeline("glsl_elementwiseAdd_comp", types);
                break;
            case BinaryOpOperation_MUL:
                mBinaryPipeline = mVkBackend->getPipeline("glsl_elementwiseMul_comp", types);
                break;
            default:
                MNN_PRINT("Not Supported Binary Operation: %d\n", mType);
                MNN_ASSERT(false);
                break;
        }

        const int iw = input0->width();
        const int ih = input0->height();

        MNN_ASSERT(input0->dimensions() == input1->dimensions());

        const int icDiv4 = UP_DIV(input0->channel(), 4);

        auto binaryOpParam = reinterpret_cast<ConstBuffer*>(mConstBuffer->map());
        ::memset(binaryOpParam, 0, sizeof(ConstBuffer));

        binaryOpParam->imgSize[0] = iw;
        binaryOpParam->imgSize[1] = ih;
        binaryOpParam->imgSize[2] = icDiv4 * input0->batch();
        binaryOpParam->imgSize[3] = 0;

        mConstBuffer->flush(true, 0, sizeof(ConstBuffer));
        mConstBuffer->unmap();

        auto sampler = mVkBackend->getCommonSampler();
        mDescriptorSet.reset(mBinaryPipeline->createSet());
        mDescriptorSet->writeImage(reinterpret_cast<VkImageView>(output->deviceId()), sampler->get(),
                                   VK_IMAGE_LAYOUT_GENERAL, 0);
        mDescriptorSet->writeImage(reinterpret_cast<VkImageView>(input0->deviceId()), sampler->get(),
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        mDescriptorSet->writeImage(reinterpret_cast<VkImageView>(input1->deviceId()), sampler->get(),
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
        mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 3, mConstBuffer->size());
        mBinaryPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(iw, 8), UP_DIV(ih, 8), UP_DIV(icDiv4 * input0->batch(), 4));
    }

    return NO_ERROR;
}

class VulkanBinaryCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new VulkanBinary(op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_BinaryOp, new VulkanBinaryCreator);
    return true;
}();

} // namespace MNN
