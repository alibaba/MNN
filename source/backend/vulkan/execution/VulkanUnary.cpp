//
//  VulkanUnary.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/execution/VulkanUnary.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

struct Param {
    ivec4 size;
    ivec4 stride;
};

VulkanUnary::VulkanUnary(const std::string& midType, Backend* bn, bool image) : VulkanBasicExecution(bn) {
    auto vkbackend = static_cast<VulkanBackend*>(bn);
    mParam         = std::make_shared<VulkanBuffer>(vkbackend->getMemoryPool(), false, sizeof(Param), nullptr,
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    std::string prefix = "glsl_unaryBuffer_";
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    if (image) {
        prefix = "glsl_unaryImage_";
        types = {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
    }
    std::string posfix = "_comp";
    // get pipeline
    mUnaryPipeline = vkbackend->getPipeline(prefix + midType + posfix, types);
    mDesSet.reset(mUnaryPipeline->createSet());
}

VulkanUnary::~VulkanUnary() {
}

static std::string _getMidType(const Op* op) {
    std::string midType = "";
    if (op->type() == OpType_TanH) {
        midType = "TANH";
    }
    else if (op->type() == OpType_Sigmoid) {
        midType = "SIGMOID";
    } else {
        // unary op
        auto unaryType = op->main_as_UnaryOp()->opType();
#define SETTYPE(type, name) if (unaryType == type) {midType = name; break;}
        do {
            SETTYPE(UnaryOpOperation_RSQRT, "RSQRT");
            SETTYPE(UnaryOpOperation_SIGN, "SIGN");
            SETTYPE(UnaryOpOperation_ABS, "ABS");
            SETTYPE(UnaryOpOperation_NEG, "NEG");
            SETTYPE(UnaryOpOperation_EXP, "EXP");
            SETTYPE(UnaryOpOperation_SQRT, "SQRT");
            SETTYPE(UnaryOpOperation_SQUARE, "SQUARE");
            SETTYPE(UnaryOpOperation_LOG, "LOG");
        } while(false);
#undef SETTYPE
    }
    return midType;
}

ErrorCode VulkanUnary::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const VulkanCommandPool::Buffer* cmdBuffer) {
    // set param
    auto size = inputs[0]->elementSize();
    auto sizeC4 = UP_DIV(size, 4);
    bool image = MNN_DATA_FORMAT_NC4HW4 == TensorUtils::getDescribe(inputs[0])->dimensionFormat;
    auto paramPtr = reinterpret_cast<Param*>(mParam->map());
    paramPtr->size[0] = sizeC4;
    if (image) {
        paramPtr->size[1] = inputs[0]->batch() * UP_DIV(inputs[0]->channel(), 4);
        paramPtr->size[2] = inputs[0]->height();
        paramPtr->size[3] = inputs[0]->width();
    }
    mParam->unmap();
    if (image) {
        auto vkBn = (VulkanBackend*)backend();
        auto inputTensor = vkBn->findTensor(inputs[0]->deviceId());
        cmdBuffer->barrierImageIfNeeded(inputTensor->image(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        // cmdBuffer->barrierImage(inputTensor->image()->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        mDesSet->writeImage((VkImageView)(outputs[0])->deviceId(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
        mDesSet->writeImage((VkImageView)(inputs[0])->deviceId(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        mDesSet->writeBuffer(mParam->buffer(), 2, mParam->size());
    } else {
        cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(inputs[0]->deviceId()), 0, inputs[0]->size());
        mDesSet->writeBuffer(reinterpret_cast<VkBuffer>(outputs[0]->deviceId()), 0, outputs[0]->size());
        mDesSet->writeBuffer(reinterpret_cast<VkBuffer>(inputs[0]->deviceId()), 1, inputs[0]->size());
        mDesSet->writeBuffer(mParam->buffer(), 2, mParam->size());
    }
    mUnaryPipeline->bind(cmdBuffer->get(), mDesSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(sizeC4, 256), 1, 1);

    return NO_ERROR;
}

class VulkanUnaryCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        bool image = MNN_DATA_FORMAT_NC4HW4 == TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        if (inputs[0]->buffer().type.code != halide_type_float) {
            return nullptr;
        }
        auto midType = _getMidType(op);
        if (midType.empty()) {
            return nullptr;
        }
        return new VulkanUnary(midType, bn, image);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_UnaryOp, new VulkanUnaryCreator);
    VulkanBackend::addCreator(OpType_TanH, new VulkanUnaryCreator);
    VulkanBackend::addCreator(OpType_Sigmoid, new VulkanUnaryCreator);
    return true;
}();

} // namespace MNN
