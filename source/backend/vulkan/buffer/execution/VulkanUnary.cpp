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
    vec4 slope;
};

VulkanUnary::VulkanUnary(const std::string& midType, Backend* bn, bool isInt, float slope0, float slope1, bool iscast) : VulkanBasicExecution(bn) {
    mSlopes[0] = slope0;
    mSlopes[1] = slope1;
    auto vkbackend = static_cast<VulkanBackend*>(bn);
    mParam         = std::make_shared<VulkanBuffer>(vkbackend->getMemoryPool(), false, sizeof(Param), nullptr,
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    auto types = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    if (iscast) {
        mUnaryPipeline = vkbackend->getPipeline(midType, types);
    } else {
        if (!midType.empty()) {
            std::string pKey = "glsl_unary_";
            if (isInt) {
                pKey += "int_";
            }
            pKey += midType;
            pKey += "_";
            if (!isInt && vkbackend->useFP16()) {
                pKey += "FP16_";
            }
            pKey += "comp";
            mUnaryPipeline = vkbackend->getPipeline(pKey, types);
        } else {
            std::string pKey = (vkbackend->useFP16() && !isInt) ? "glsl_unary_FP16_comp" : "glsl_unary_comp";
            mUnaryPipeline = vkbackend->getPipeline(pKey, types);
        }
    }
    mDesSet.reset(mUnaryPipeline->createSet());
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
            SETTYPE(UnaryOpOperation_GELU, "GELU");
            // Since SPIR-V lacks a built-in erf (gauss error function) instruction and the existing shader implementation of GELU is essentially an approximation of erf, there is no need to add a new implementation of GELU_STANDARD.
            SETTYPE(UnaryOpOperation_GELU_STANDARD, "GELU");
            SETTYPE(UnaryOpOperation_SILU, "SILU");

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
    auto size = inputs[0]->elementSize();
    auto sizeC4 = UP_DIV(size, 4);
    auto paramPtr = reinterpret_cast<Param*>(mParam->map());
    paramPtr->size[0] = sizeC4;
    ::memcpy(paramPtr->slope, mSlopes, sizeof(float) * 4);
    mParam->unmap();
    auto vkBn = (VulkanBackend*)backend();
    auto inputTensor = vkBn->getBuffer(inputs[0]);
    auto outputTensor = vkBn->getBuffer(outputs[0]);
    mDesSet->writeBuffer(outputTensor, 0);
    mDesSet->writeBuffer(inputTensor, 1);
    mDesSet->writeBuffer(mParam->buffer(), 2, mParam->size());
    mUnaryPipeline->bind(cmdBuffer->get(), mDesSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(sizeC4, 256), 1, 1);

    return NO_ERROR;
}

class VulkanUnaryCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        auto vkBn = static_cast<VulkanBackend *>(bn);
        if (op->type() == OpType_ReLU6) {
            float minValue = 0.0f;
            float maxValue = 6.0f;
            if (op->main_as_Relu6() != nullptr) {
                minValue = op->main_as_Relu6()->minValue();
                maxValue = op->main_as_Relu6()->maxValue();
            }
            return new VulkanUnary("CLAMP", bn, false, minValue, maxValue);
        }
        if (op->type() == OpType_ReLU) {
            return new VulkanUnary("RELU", bn, false, op->main_as_Relu()->slope());
        }
        if (op->type() == OpType_Cast) {
            if (inputs[0]->getType().bytes() != 4 || outputs[0]->getType().bytes() != 4) {
                return nullptr;
            }
            if (op->main_as_CastParam()->dstT() == MNN::DataType_DT_BOOL) {
                return new VulkanUnary("glsl_cast_int_bool_comp", bn, false, 0.0f, 0.0f, true);
            }

            auto srcCode = inputs[0]->getType().code;
            auto dstCode = outputs[0]->getType().code;

            if (srcCode == dstCode) {
                if (srcCode == halide_type_float || srcCode == halide_type_int) {
                    return new VulkanUnary("", bn, srcCode == halide_type_int);
                }
                return nullptr;
            }

            std::string pKey;
            if (srcCode == halide_type_float && dstCode == halide_type_int) {
                pKey = "glsl_cast_float_int_";
            } else if (srcCode == halide_type_int && dstCode == halide_type_float) {
                pKey = "glsl_cast_float_int_REVERT_";
            } else {
                return nullptr;
            }

            if (vkBn->useFP16()) {
                pKey += "FP16_";
            }
            pKey += "comp";
            return new VulkanUnary(pKey, bn, false, 0.0f, 0.0f, true);
        }
        auto midType = _getMidType(op);
        if (midType.empty()) {
            return nullptr;
        }
        return new VulkanUnary(midType, bn, inputs[0]->getType().code == halide_type_int);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_ReLU6, new VulkanUnaryCreator);
    VulkanBackend::addCreator(OpType_ReLU, new VulkanUnaryCreator);
    VulkanBackend::addCreator(OpType_Cast, new VulkanUnaryCreator);
    VulkanBackend::addCreator(OpType_UnaryOp, new VulkanUnaryCreator);
    VulkanBackend::addCreator(OpType_TanH, new VulkanUnaryCreator);
    VulkanBackend::addCreator(OpType_Sigmoid, new VulkanUnaryCreator);
    return true;
}();

} // namespace MNN
