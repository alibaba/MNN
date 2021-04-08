//
//  VulkanUnary.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanUnary.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

struct Param {
    ivec4 size;
};

VulkanUnary::VulkanUnary(const std::string& midType, Backend* bn, bool image) : VulkanBasicExecution(bn) {
    auto vkbackend = static_cast<VulkanBackend*>(bn);
    auto prefix = "glsl_unaryImage_";
    auto types = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    std::string posfix = "_comp";
    // get pipeline
    mUnaryPipeline = vkbackend->getPipeline(prefix + midType + posfix, types);
}

VulkanUnary::~VulkanUnary() {
}

static std::string _getMidType(const Op* op) {
    std::string midType = "";
    if (op->type() == OpType_TanH) {
        midType = "TANH";
    } else if (op->type() == OpType_Sigmoid) {
        midType = "SIGMOID";
    } else {
        // unary op
        auto unaryType = op->main_as_UnaryOp()->opType();
#define SETTYPE(type, name) if (unaryType == type) {midType = name; break;}
        do {
            SETTYPE(UnaryOpOperation_SIGMOID, "SIGMOID");
            SETTYPE(UnaryOpOperation_TANH, "TANH");
            SETTYPE(UnaryOpOperation_RSQRT, "RSQRT");
            SETTYPE(UnaryOpOperation_SIGN, "SIGN");
            SETTYPE(UnaryOpOperation_ABS, "ABS");
            SETTYPE(UnaryOpOperation_NEG, "NEG");
            SETTYPE(UnaryOpOperation_EXP, "EXP");
            SETTYPE(UnaryOpOperation_SQRT, "SQRT");
            SETTYPE(UnaryOpOperation_SQUARE, "SQUARE");
            SETTYPE(UnaryOpOperation_LOG, "LOG");
            SETTYPE(UnaryOpOperation_TAN, "TAN");
            SETTYPE(UnaryOpOperation_COS, "COS");
            SETTYPE(UnaryOpOperation_SIN, "SIN");
            SETTYPE(UnaryOpOperation_CEIL, "CEIL");
            SETTYPE(UnaryOpOperation_FLOOR, "FLOOR");
            SETTYPE(UnaryOpOperation_EXPM1, "EXPM1");
            SETTYPE(UnaryOpOperation_RECIPROCAL, "RECIPROCAL");

            SETTYPE(UnaryOpOperation_SINH, "SINH");
            SETTYPE(UnaryOpOperation_ASIN, "ASIN");
            SETTYPE(UnaryOpOperation_ASINH, "ASINH");
            SETTYPE(UnaryOpOperation_COSH, "COSH");
            SETTYPE(UnaryOpOperation_ACOS, "ACOS");
            SETTYPE(UnaryOpOperation_ACOSH, "ACOSH");
            SETTYPE(UnaryOpOperation_ATAN, "ATAN");
            SETTYPE(UnaryOpOperation_ATANH, "ATANH");
            SETTYPE(UnaryOpOperation_LOG1P, "LOG1P");
            
            SETTYPE(UnaryOpOperation_ROUND, "ROUND");
            SETTYPE(UnaryOpOperation_HARDSWISH, "HARDSWISH");
        } while(false);
#undef SETTYPE
    }
    return midType;
}

ErrorCode VulkanUnary::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const VulkanCommandPool::Buffer* cmdBuffer) {
    // set param
    auto vkbackend = static_cast<VulkanBackend*>(backend());
    auto inputTensor = (VulkanTensor*)(inputs[0]->deviceId());
    auto outputTensor = (VulkanTensor*)(outputs[0]->deviceId());
    mDesSet.resize(inputTensor->imageSize());
    auto needSize = sizeof(Param);
    if (needSize < vkbackend->proty().limits.nonCoherentAtomSize) {
        needSize = vkbackend->proty().limits.nonCoherentAtomSize;
    }
    mParam = std::make_shared<VulkanBuffer>(vkbackend->getMemoryPool(), false, needSize * inputTensor->imageSize(), nullptr,
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    auto paramOrigin = (Param*)mParam->map();
    for (int n=0; n<inputTensor->imageSize(); ++n) {
        auto paramPtr = (Param*)((uint8_t*)paramOrigin + n * needSize);
        mDesSet[n].reset(mUnaryPipeline->createSet());
        auto inputT = inputTensor->image(n);
        auto outputT = outputTensor->image(n);
        auto totalSize = inputT->depth() * inputT->height() * inputT->width();
        paramPtr->size[0] = inputT->depth() * inputT->height() * inputT->width();
        paramPtr->size[1] = inputT->depth();
        paramPtr->size[2] = inputT->height();
        paramPtr->size[3] = inputT->width();
        auto vkBn = (VulkanBackend*)backend();
        cmdBuffer->barrierImage(inputT->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        mDesSet[n]->writeImage(outputT->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
        mDesSet[n]->writeImage(inputT->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        mDesSet[n]->writeBuffer(mParam->buffer(), 2, sizeof(Param), n * needSize);
        mUnaryPipeline->bind(cmdBuffer->get(), mDesSet[n]->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalSize, 256), 1, 1);
    }
    mParam->unmap();
    return NO_ERROR;
}

class VulkanUnaryCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        if (inputs[0]->buffer().type.code != halide_type_float) {
            return nullptr;
        }
        auto midType = _getMidType(op);
        if (midType.empty()) {
            return nullptr;
        }
        return new VulkanUnary(midType, bn, true);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_UnaryOp, new VulkanUnaryCreator);
    VulkanBackend::addCreator(OpType_TanH, new VulkanUnaryCreator);
    VulkanBackend::addCreator(OpType_Sigmoid, new VulkanUnaryCreator);
    return true;
}();

} // namespace MNN
