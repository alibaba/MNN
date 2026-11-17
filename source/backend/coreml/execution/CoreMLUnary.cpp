//
//  CoreMLUnary.cpp
//  MNN
//
//  Created by MNN on 2021/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLUnary.hpp"

namespace MNN {

CoreMLUnary::CoreMLUnary(MNN::Backend* b, const MNN::Op* op, const std::vector<Tensor*>& inputs,
                         const std::vector<MNN::Tensor*>& outputs)
    : CoreMLCommonExecution(b, op) {
    switch (op->main_as_UnaryOp()->opType()) {
        case UnaryOpOperation_ABS:
        case UnaryOpOperation_ACOS:
        case UnaryOpOperation_ACOSH:
        case UnaryOpOperation_ASIN:
        case UnaryOpOperation_ASINH:
        case UnaryOpOperation_ATAN:
        case UnaryOpOperation_ATANH:
        case UnaryOpOperation_CEIL:
        case UnaryOpOperation_COS:
        case UnaryOpOperation_COSH:
        case UnaryOpOperation_ERF:
        case UnaryOpOperation_EXP:
        case UnaryOpOperation_FLOOR:
        case UnaryOpOperation_GELU:
        case UnaryOpOperation_GELU_STANDARD:
        case UnaryOpOperation_HARDSWISH:
        case UnaryOpOperation_LOG:
        case UnaryOpOperation_LOG1P:
        case UnaryOpOperation_NEG:
        case UnaryOpOperation_RECIPROCAL:
        case UnaryOpOperation_ROUND:
        case UnaryOpOperation_RSQRT:
        case UnaryOpOperation_SIGMOID:
        case UnaryOpOperation_SIGN:
        case UnaryOpOperation_SILU:
        case UnaryOpOperation_SIN:
        case UnaryOpOperation_SINH:
        case UnaryOpOperation_SQRT:
        case UnaryOpOperation_SQUARE:
        case UnaryOpOperation_TAN:
        case UnaryOpOperation_TANH:
        case UnaryOpOperation_BNLL:
        case UnaryOpOperation_EXPM1:
        case UnaryOpOperation_ERFC:
            break;
        default:
            mValid = false;
            break;
    }
}

ErrorCode CoreMLUnary::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    auto inputName = mCoreMLBackend->getTensorName(inputs[0]);
    auto opType = mOp->main_as_UnaryOp()->opType();
    std::map<UnaryOpOperation, std::string> unaryTypeMap = {{UnaryOpOperation_ABS, "abs"},
                                                            {UnaryOpOperation_EXP, "exp"},
                                                            {UnaryOpOperation_SQRT, "sqrt"},
                                                            {UnaryOpOperation_RSQRT, "rsqrt"},
                                                            {UnaryOpOperation_LOG, "log"},
                                                            {UnaryOpOperation_RECIPROCAL, "inverse"},
                                                            {UnaryOpOperation_SIN, "sin"},
                                                            {UnaryOpOperation_ASIN, "asin"},
                                                            {UnaryOpOperation_SINH, "sinh"},
                                                            {UnaryOpOperation_ASINH, "asinh"},
                                                            {UnaryOpOperation_COS, "cos"},
                                                            {UnaryOpOperation_ACOS, "acos"},
                                                            {UnaryOpOperation_COSH, "cosh"},
                                                            {UnaryOpOperation_ACOSH, "acosh"},
                                                            {UnaryOpOperation_TAN, "tan"},
                                                            {UnaryOpOperation_ATAN, "atan"},
                                                            {UnaryOpOperation_TANH, "tanh"},
                                                            {UnaryOpOperation_ATANH, "atanh"},
                                                            {UnaryOpOperation_ERF, "erf"},
                                                            {UnaryOpOperation_CEIL, "ceil"},
                                                            {UnaryOpOperation_FLOOR, "floor"},
                                                            {UnaryOpOperation_ROUND, "round"},
                                                            {UnaryOpOperation_SIGN, "sign"},
                                                            {UnaryOpOperation_SIGMOID, "sigmoid"},
                                                            {UnaryOpOperation_LOG1P, "log1p"},
                                                            {UnaryOpOperation_SQUARE, "square"},
                                                            {UnaryOpOperation_HARDSWISH, "hardswish"},
                                                            {UnaryOpOperation_SILU, "silu"},
                                                            {UnaryOpOperation_GELU, "gelu"},
                                                            {UnaryOpOperation_GELU_STANDARD, "gelu"},
                                                            {UnaryOpOperation_EXPM1, "expm1"},
                                                            {UnaryOpOperation_ERFC, "erfc"},
                                                            {UnaryOpOperation_BNLL, "softplus"},
                                                            {UnaryOpOperation_ERFINV, "erfinv"}};
    auto unary = [&](const std::string& type, const std::string& x, const std::string& tag) {
        auto b = operation(type);
        b.input("x", x);
        if (type == "log") {
            b.scalar("epsilon", 0.0f);
        }
        return b.output(inputs[0]->shape(), tag);
    };
    auto binary = [&](const std::string& type, const std::string& x, const std::string& y, const std::string& tag) {
        return operation(type).input("x", x).input("y", y).output(inputs[0]->shape(), tag);
    };
    if (opType == UnaryOpOperation_EXPM1 || opType == UnaryOpOperation_ERFC || opType == UnaryOpOperation_LOG1P) {
        const auto fp32 = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32;
        const auto shape = inputs[0]->shape();
        int serial = 0;
        auto constant = [&](float value) {
            return mCoreMLBackend->buildScalarFloat32(value, mName + "_stable_const_" + std::to_string(serial++));
        };
        auto calc = [&](const char* type, const std::string& x, const std::string& y) {
            return operation(type).input("x", x).input("y", y).output(shape, "stable_" + std::to_string(serial++),
                                                                      fp32);
        };
        auto apply = [&](const char* type, const std::string& x) {
            auto op = operation(type);
            op.input("x", x);
            if (std::string(type) == "log") {
                op.scalarFloat32("epsilon", 0.0f);
            }
            return op.output(shape, "stable_" + std::to_string(serial++), fp32);
        };
        auto x = operation("cast")
                     .input("x", inputName)
                     .scalar("dtype", "fp32", "stable_dtype")
                     .output(shape, "stable_input", fp32);
        auto one = constant(1.0f);
        auto magnitude = apply("abs", x);
        std::string result;
        if (opType == UnaryOpOperation_ERFC) {
            // Evaluate the complementary tail directly, avoiding subtraction from a rounded erf.
            auto t = calc("real_div", one, calc("add", one, calc("mul", constant(0.5f), magnitude)));
            const float coefficients[] = {0.17087277f,  -0.82215223f, 1.48851587f, -1.13520398f, 0.27886807f,
                                          -0.18628806f, 0.09678418f,  0.37409196f, 1.00002368f};
            auto polynomial = constant(coefficients[0]);
            for (int i = 1; i < 9; ++i) {
                polynomial = calc("add", calc("mul", t, polynomial), constant(coefficients[i]));
            }
            auto exponent = calc("sub", calc("mul", t, polynomial),
                                 calc("add", calc("mul", magnitude, magnitude), constant(1.26551223f)));
            auto tail = calc("mul", t, apply("exp", exponent));
            auto negative = operation("less")
                                .input("x", x)
                                .input("y", constant(0.0f))
                                .output(shape, "negative", CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__BOOL);
            result = operation("select")
                         .input("cond", negative)
                         .input("a", calc("sub", constant(2.0f), tail))
                         .input("b", tail)
                         .output(shape, "erfc_stable", fp32);
        } else {
            const bool logarithm = opType == UnaryOpOperation_LOG1P;
            const float expCoefficients[] = {1.0f / 720.0f, 1.0f / 120.0f, 1.0f / 24.0f, 1.0f / 6.0f, 0.5f, 1.0f};
            const float logCoefficients[] = {-1.0f / 6.0f, 0.2f, -0.25f, 1.0f / 3.0f, -0.5f, 1.0f};
            auto coefficients = logarithm ? logCoefficients : expCoefficients;
            auto polynomial = constant(coefficients[0]);
            for (int i = 1; i < 6; ++i) {
                polynomial = calc("add", calc("mul", x, polynomial), constant(coefficients[i]));
            }
            auto small = calc("mul", x, polynomial);
            auto general = logarithm ? apply("log", calc("add", x, one)) : calc("sub", apply("exp", x), one);
            auto predicate = operation("less")
                                 .input("x", magnitude)
                                 .input("y", constant(0.125f))
                                 .output(shape, "small_argument", CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__BOOL);
            result = operation("select")
                         .input("cond", predicate)
                         .input("a", small)
                         .input("b", general)
                         .output(shape, "stable_result", fp32);
        }
        operation("cast").input("x", result).scalar("dtype", "fp16", "narrow_dtype").output(outputs);
        return NO_ERROR;
    }
    if (opType == UnaryOpOperation_ROUND) {
        // MNN rounds halfway away from zero; MIL round uses ties to even.
        auto magnitude = unary("abs", inputName, "magnitude");
        auto integral = unary("floor", magnitude, "integral");
        auto fraction = binary("sub", magnitude, integral, "fraction");
        auto increment = operation("greater_equal")
                             .input("x", fraction)
                             .scalar("y", 0.5f, "half")
                             .output(inputs[0]->shape(), "increment", CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__BOOL);
        auto next = binary("add", integral, buildScalar(1.0f, "one"), "next");
        auto rounded = operation("select")
                           .input("cond", increment)
                           .input("a", next)
                           .input("b", integral)
                           .output(inputs[0]->shape(), "rounded");
        auto sign = unary("sign", inputName, "sign");
        operation("mul").input("x", rounded).input("y", sign).output(outputs);
        return NO_ERROR;
    }
    if (opType == UnaryOpOperation_HARDSWISH) {
        auto gate = operation("sigmoid_hard")
                        .input("x", inputName)
                        .scalar("alpha", 1.0f / 6.0f, "sixth")
                        .scalar("beta", 0.5f, "half")
                        .output(inputs[0]->shape(), "gate");
        operation("mul").input("x", inputName).input("y", gate).output(outputs);
        return NO_ERROR;
    }
    if (opType == UnaryOpOperation_SILU) {
        auto gate = operation("sigmoid").input("x", inputName).output(inputs[0]->shape(), "sigmoid");
        operation("mul").input("x", inputName).input("y", gate).output(outputs);
        return NO_ERROR;
    }
    auto type = unaryTypeMap.find(opType);
    if (opType != UnaryOpOperation_NEG && type == unaryTypeMap.end()) {
        return NOT_SUPPORT;
    }
    auto builder = operation(opType == UnaryOpOperation_NEG ? "mul" : type->second);
    builder.input("x", inputs[0]);
    switch (opType) {
        case UnaryOpOperation_RECIPROCAL:
        case UnaryOpOperation_RSQRT:
        case UnaryOpOperation_LOG:
            builder.scalar<float>("epsilon", 0.0f);
            break;
        case UnaryOpOperation_LOG1P:
            break;
        case UnaryOpOperation_NEG:
            if (inputs[0]->getType() == halide_type_of<int>()) {
                builder.scalar<int>("y", -1, "neg");
            } else {
                builder.scalar<float>("y", -1.0f, "neg");
            }
            break;
        case UnaryOpOperation_HARDSWISH:
        case UnaryOpOperation_SILU:
            break;
        case UnaryOpOperation_GELU:
        case UnaryOpOperation_GELU_STANDARD:
            // EXACT, TANH_APPROXIMATION, SIGMOID_APPROXIMATION
            builder.scalar<std::string>("mode", "EXACT");
            break;
        default:
            // MNN_ERROR("CoreMLUnary not support %s\n", MNN::EnumNameUnaryOpOperation(opType));
            break;
    }
    builder.output(outputs);
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLUnary, OpType_UnaryOp);
} // namespace MNN
