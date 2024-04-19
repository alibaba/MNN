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
    int activationType = 0;
};
std::string VulkanBinary::getMidName(const Op *op) {
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
            case BinaryOpOperation_ATAN2:
                mid = "ATAN2";
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
            case BinaryOpOperation_LESS:
                mid = "LESS";
                break;
            case BinaryOpOperation_LESS_EQUAL:
                mid = "LESSEQUAL";
                break;
            case BinaryOpOperation_GREATER:
                mid = "GREATER";
                break;
            case BinaryOpOperation_GREATER_EQUAL:
                mid = "GREATEREQUAL";
                break;
            case BinaryOpOperation_EQUAL:
                mid = "EQUAL";
                break;
            case BinaryOpOperation_NOTEQUAL:
                mid = "NOTEQUAL";
                break;
            case BinaryOpOperation_MOD:
                mid = "VMOD";
                break;
            case BinaryOpOperation_FLOORDIV:
                mid = "FLOORDIV";
                break;
            case BinaryOpOperation_FLOORMOD:
                mid = "FLOORMOD";
                break;
            default:
                FUNC_PRINT(op->main_as_BinaryOp()->opType());
                break;
        }
    }
    return mid;
}
static std::string _getShaderName(const Op* op, bool isInt) {
    std::string prefix = "glsl_binary_";
    if (isInt) {
        prefix = "glsl_binary_int_";
    }
    std::string posfix = "_comp";
    auto mid = VulkanBinary::getMidName(op);
    if (mid.empty()) {
        return mid;
    }
    return prefix + mid + posfix;
}

VulkanBinary::VulkanBinary(const std::string& shaderName, Backend* bn, int activationType, int inputSize) : VulkanBasicExecution(bn) {
    MNN_ASSERT(inputSize >= 2);
    auto vkBn   = static_cast<VulkanBackend*>(bn);
    mBinaryPipeline = vkBn->getPipeline(shaderName, {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    });
    mActivationType = activationType;
    mConstBuffer.resize(inputSize - 1);
    mDescriptorSet.resize(inputSize - 1);
    for (int i=0; i<mConstBuffer.size(); ++i) {
        mConstBuffer[i] = vkBn->allocUniform();
        mDescriptorSet[i].reset(mBinaryPipeline->createSet());
    }
}

VulkanBinary::~VulkanBinary() {
    auto vkBn   = static_cast<VulkanBackend*>(backend());
    for (auto buffer : mConstBuffer) {
        vkBn->recycleUniform(buffer);
    }
}

ErrorCode VulkanBinary::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    MNN_ASSERT(1 == outputs.size());

    auto vkBn = (VulkanBackend*)backend();
    auto input0DataCount = TensorUtils::getRawSize(inputs[0]);
    auto input1DataCount = TensorUtils::getRawSize(inputs[1]);

    auto input0Scalar = input0DataCount == 1;
    auto input1Scalar = input1DataCount == 1;
    auto writeBinary = [&](const VULKAN_TENSOR& input0, const VULKAN_TENSOR& input1, const VULKAN_TENSOR& output, int index) {
        auto constBuffer = mConstBuffer[index];
        auto total = std::get<1>(output) / 4 / sizeof(float);
        auto binaryOpParam = reinterpret_cast<ConstBuffer*>(constBuffer->map());
        ::memset(binaryOpParam, 0, sizeof(ConstBuffer));
        binaryOpParam->stride00[3] = total;
        binaryOpParam->stride00[0] = 1;
        binaryOpParam->stride00[1] = 1;
        if (input0Scalar) {
            binaryOpParam->stride00[0] = 0;
        }
        if (input1Scalar) {
            binaryOpParam->stride00[1] = 0;
        }
        binaryOpParam->activationType = mActivationType;
        constBuffer->unmap();
        std::shared_ptr<VulkanLayout::DescriptorSet> desSet = mDescriptorSet[index];
        desSet->writeBuffer(output, 0);
        desSet->writeBuffer(input0, 1);
        desSet->writeBuffer(input1, 2);
        cmdBuffer->barrierSource(input0);
        cmdBuffer->barrierSource(input1);
        desSet->writeBuffer(constBuffer->buffer(), 3, constBuffer->size());
        mBinaryPipeline->bind(cmdBuffer->get(), desSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(total, 256), 1, 1);
    };
    auto input0T = vkBn->getBuffer(inputs[0]);
    auto input1T = vkBn->getBuffer(inputs[1]);
    auto outputT = vkBn->getBuffer(outputs[0]);
    writeBinary(input0T, input1T, outputT, 0);
    if (inputs.size() > 2) {
        for (int i=2; i<inputs.size(); ++i) {
            writeBinary(vkBn->getBuffer(outputs[0]), vkBn->getBuffer(inputs[i]), vkBn->getBuffer(outputs[0]), i-1);
        }
    }
    return NO_ERROR;
}

class VulkanBinaryCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        auto input0 = inputs[0];
        auto shader = _getShaderName(op, input0->getType().code == halide_type_int);
        if (shader.empty()) {
            return nullptr;
        }
        int activationType = 0;
        if (op->type() == OpType_BinaryOp) {
            activationType = op->main_as_BinaryOp()->activationType();
        }
        return new VulkanBinary(shader, backend, activationType, inputs.size());
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_BinaryOp, new VulkanBinaryCreator);
    VulkanBackend::addCreator(OpType_Eltwise, new VulkanBinaryCreator);
    return true;
}();

} // namespace MNN
