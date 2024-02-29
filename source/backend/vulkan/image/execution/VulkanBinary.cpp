//
//  VulkanBinary.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanBinary.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {

struct ConstBuffer {
    ivec4 stride00;
    ivec4 posLimit;
    int activationType;
};
static std::string _getShaderName(const Op* op, bool image) {
    std::string prefix = "glsl_binaryImage_";
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
            case BinaryOpOperation_POW:
                mid = "POW";
                break;
            case BinaryOpOperation_SquaredDifference:
                mid = "SQUDIFF";
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

VulkanBinary::VulkanBinary(const std::string& shaderName, Backend* bn, bool image, int number, int activationType) : VulkanBasicExecution(bn) {
    auto vkBn   = static_cast<VulkanBackend*>(bn);
    mBinaryPipeline = vkBn->getPipeline(shaderName, {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    });
    mActivationType = activationType;
}

VulkanBinary::~VulkanBinary() {
}

ErrorCode VulkanBinary::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    MNN_ASSERT(1 == outputs.size());
    auto input0T = (VulkanTensor*)(inputs[0]->deviceId());
    auto input1T = (VulkanTensor*)(inputs[1]->deviceId());
    auto outputT = (VulkanTensor*)(outputs[0]->deviceId());
    auto vkBn = (VulkanBackend*)backend();

    int number = outputT->imageSize() * ((int)inputs.size() - 1);
    if (mConstBuffer.size() != number) {
        mConstBuffer.resize(number);
        for (int i=0; i<number; ++i) {
            mConstBuffer[i] = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, sizeof(ConstBuffer), nullptr,
                                                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        }
        mDescriptorSet.resize(number);
        for (int i=0; i<number; ++i) {
            mDescriptorSet[i].reset(mBinaryPipeline->createSet());
        }
    }
    auto input0DataCount = TensorUtils::getRawSize(inputs[0]);
    auto input1DataCount = TensorUtils::getRawSize(inputs[1]);

    auto input0Scalar = input0DataCount == 1;
    auto input1Scalar = input1DataCount == 1;
    auto writeBinary = [&](VulkanTensor* input0T, VulkanTensor* input1T, VulkanTensor* outputT, int tensorIndex) {
        auto imageSize = outputT->imageSize();
        for (int index=0; index < imageSize; ++index) {
            auto input0 = input0T->image(index % input0T->imageSize());
            auto input1 = input1T->image(index % input1T->imageSize());
            auto output = outputT->image(index);
            auto total = output->width() * output->height();
            auto constBuffer = mConstBuffer[tensorIndex * imageSize + index];
            auto binaryOpParam = reinterpret_cast<ConstBuffer*>(constBuffer->map());
            ::memset(binaryOpParam, 0, sizeof(ConstBuffer));
            binaryOpParam->stride00[3] = total;
            binaryOpParam->stride00[0] = output->width();
            binaryOpParam->stride00[1] = output->height();
            binaryOpParam->stride00[2] = 0;
            binaryOpParam->posLimit[0] = 1;
            binaryOpParam->posLimit[1] = 1;
            binaryOpParam->activationType = mActivationType;
            if (input0Scalar) {
                binaryOpParam->posLimit[0] = 0;
            }
            if (input1Scalar) {
                binaryOpParam->posLimit[1] = 0;
            }
            constBuffer->unmap();
            std::shared_ptr<VulkanLayout::DescriptorSet> desSet = mDescriptorSet[tensorIndex * imageSize + index];
            auto sampler = vkBn->getCommonSampler(true);
            desSet->writeImage(output->view(), sampler->get(),
                                       VK_IMAGE_LAYOUT_GENERAL, 0);
            input0->barrierRead(cmdBuffer->get());
            input1->barrierRead(cmdBuffer->get());
            output->barrierWrite(cmdBuffer->get());
            desSet->writeImage(input0->view(), sampler->get(),
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
            desSet->writeImage(input1->view(), sampler->get(),
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
            desSet->writeBuffer(constBuffer->buffer(), 3, constBuffer->size());
            mBinaryPipeline->bind(cmdBuffer->get(), desSet->get());
            vkCmdDispatch(cmdBuffer->get(), UP_DIV(total, 256), 1, 1);
        }
    };
    writeBinary(input0T, input1T, outputT, 0);
    if (inputs.size() > 2) {
        for (int i=2; i<inputs.size(); ++i) {
            writeBinary(reinterpret_cast<VulkanTensor*>(outputs[0]->deviceId()), reinterpret_cast<VulkanTensor*>(inputs[i]->deviceId()), reinterpret_cast<VulkanTensor*>(outputs[0]->deviceId()), i-1);
        }
    }
    return NO_ERROR;
}

class VulkanBinaryCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        auto input0 = inputs[0];
        auto image = TensorUtils::getDescribe(input0)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
        auto shader = _getShaderName(op, image);
        if (shader.empty()) {
            return nullptr;
        }
        int activationType = 0;
        if(op->type() == OpType_BinaryOp) {
            activationType = op->main_as_BinaryOp()->activationType();
        } 
        return new VulkanBinary(shader, backend, image, (int)inputs.size() - 1, activationType);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_BinaryOp, new VulkanBinaryCreator);
    VulkanBackend::addCreator(OpType_Eltwise, new VulkanBinaryCreator);
    return true;
}();

} // namespace MNN
