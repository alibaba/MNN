//
//  CoreMLBinary.cpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLBinary.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

CoreMLBinary::CoreMLBinary(MNN::Backend* b, const MNN::Op* op, const std::vector<Tensor*>& inputs,
                           const std::vector<MNN::Tensor*>& outputs)
    : CoreMLCommonExecution(b, op) {
    if (inputs.size() != 2 || outputs.size() != 1) {
        mValid = false;
        return;
    }
    if (op->type() == OpType_BinaryOp) {
        auto type = op->main_as_BinaryOp()->opType();
        switch (type) {
            case BinaryOpOperation_ADD:
            case BinaryOpOperation_SUB:
            case BinaryOpOperation_MUL:
            case BinaryOpOperation_DIV:
            case BinaryOpOperation_POW:
            case BinaryOpOperation_REALDIV:
            case BinaryOpOperation_MINIMUM:
            case BinaryOpOperation_MAXIMUM:
            case BinaryOpOperation_GREATER:
            case BinaryOpOperation_GREATER_EQUAL:
            case BinaryOpOperation_LESS:
            case BinaryOpOperation_FLOORDIV:
            case BinaryOpOperation_SquaredDifference:
            case BinaryOpOperation_LESS_EQUAL:
            case BinaryOpOperation_FLOORMOD:
            case BinaryOpOperation_EQUAL:
            case BinaryOpOperation_MOD:
            case BinaryOpOperation_ATAN2:
            case BinaryOpOperation_LOGICALOR:
            case BinaryOpOperation_NOTEQUAL:
            case BinaryOpOperation_LOGICALXOR:
            case BinaryOpOperation_MUL_SILU:
                mValid = op->main_as_BinaryOp()->activationType() >= 0 && op->main_as_BinaryOp()->activationType() <= 1;
                break;
            default:
                mValid = false;
                break;
        }
    }
}

ErrorCode CoreMLBinary::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(inputs.size() == 2 && outputs.size() == 1);
    BinaryOpOperation binaryType;
    if (mOp->type() == OpType_BinaryOp) {
        binaryType = static_cast<BinaryOpOperation>(mOp->main_as_BinaryOp()->opType());
    } else if (mOp->type() == OpType_Eltwise) {
        auto elemType = mOp->main_as_Eltwise()->type();
        switch (elemType) {
            case EltwiseType_PROD:
                binaryType = BinaryOpOperation_MUL;
                break;
            case EltwiseType_SUM:
                binaryType = BinaryOpOperation_ADD;
                break;
            case EltwiseType_SUB:
                binaryType = BinaryOpOperation_SUB;
                break;
            case EltwiseType_MAXIMUM:
                binaryType = BinaryOpOperation_MAXIMUM;
                break;
        }
    }
    int activationType = 0;
    if (mOp->type() == OpType_BinaryOp) {
        activationType = mOp->main_as_BinaryOp()->activationType();
    }
    std::map<BinaryOpOperation, std::string> binaryTypeMap = {{BinaryOpOperation_ADD, "add"},
                                                              {BinaryOpOperation_SUB, "sub"},
                                                              {BinaryOpOperation_MUL, "mul"},
                                                              {BinaryOpOperation_DIV, "real_div"},
                                                              {BinaryOpOperation_POW, "pow"},
                                                              {BinaryOpOperation_REALDIV, "real_div"},
                                                              {BinaryOpOperation_MINIMUM, "minimum"},
                                                              {BinaryOpOperation_MAXIMUM, "maximum"},
                                                              {BinaryOpOperation_GREATER, "greater"},
                                                              {BinaryOpOperation_GREATER_EQUAL, "greater_equal"},
                                                              {BinaryOpOperation_LESS, "less"},
                                                              {BinaryOpOperation_FLOORDIV, "floor_div"},
                                                              {BinaryOpOperation_SquaredDifference, ""},
                                                              {BinaryOpOperation_LESS_EQUAL, "less_equal"},
                                                              {BinaryOpOperation_FLOORMOD, ""},
                                                              {BinaryOpOperation_EQUAL, "equal"},
                                                              {BinaryOpOperation_MOD, "mod"},
                                                              {BinaryOpOperation_ATAN2, ""},
                                                              {BinaryOpOperation_LOGICALOR, "logical_or"},
                                                              {BinaryOpOperation_NOTEQUAL, "not_equal"},
                                                              {BinaryOpOperation_BITWISE_AND, ""},
                                                              {BinaryOpOperation_BITWISE_OR, ""},
                                                              {BinaryOpOperation_BITWISE_XOR, ""},
                                                              {BinaryOpOperation_LOGICALXOR, "logical_xor"},
                                                              {BinaryOpOperation_LEFTSHIFT, ""},
                                                              {BinaryOpOperation_RIGHTSHIFT, ""}};

    if (binaryType == BinaryOpOperation_MUL_SILU) {
        const auto fp32 = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32;
        std::vector<std::string> values;
        for (int i = 0; i < 2; ++i) {
            values.push_back(operation("cast")
                                 .input("x", inputs[i])
                                 .scalar("dtype", "fp32", "accum_dtype")
                                 .output(inputs[i]->shape(), "float_" + std::to_string(i), fp32));
        }
        auto gate = operation("sigmoid").input("x", values[1]).output(inputs[1]->shape(), "sigmoid", fp32);
        gate = operation("mul").input("x", values[1]).input("y", gate).output(inputs[1]->shape(), "silu", fp32);
        auto result =
            operation("mul").input("x", values[0]).input("y", gate).output(outputs[0]->shape(), "product", fp32);
        operation("cast").input("x", result).scalar("dtype", "fp16", "output_dtype").output(outputs);
        return NO_ERROR;
    }
    const auto fp16 = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT16;
    const auto int32 = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT32;
    const auto boolean = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__BOOL;
    auto dtype = outputs[0]->getType() == halide_type_of<int>() ? int32 : fp16;
    auto shape = outputs[0]->shape();
    auto x = mCoreMLBackend->getTensorName(inputs[0]);
    auto y = mCoreMLBackend->getTensorName(inputs[1]);
    int serial = 0;
    auto binary = [&](const std::string& type, const std::string& lhs, const std::string& rhs,
                      CoreML__Specification__MILSpec__DataType resultType) {
        return operation(type).input("x", lhs).input("y", rhs).output(shape, std::to_string(serial++), resultType);
    };
    auto unary = [&](const std::string& type, const std::string& value) {
        return operation(type).input("x", value).output(shape, std::to_string(serial++), dtype);
    };
    auto select = [&](const std::string& cond, const std::string& a, const std::string& b) {
        return operation("select")
            .input("cond", cond)
            .input("a", a)
            .input("b", b)
            .output(shape, std::to_string(serial++), dtype);
    };
    if (binaryType == BinaryOpOperation_ATAN2 || binaryType == BinaryOpOperation_LOGICALOR ||
        binaryType == BinaryOpOperation_LOGICALXOR || (binaryType == BinaryOpOperation_DIV && dtype == int32)) {
        // Intermediate predicates must use the same broadcast shape as the result.
        for (int i = 0; i < 2; ++i) {
            auto& value = i == 0 ? x : y;
            if (inputs[i]->shape() == shape) {
                continue;
            }
            auto tag = "broadcast_" + std::to_string(i);
            std::vector<int> expanded(shape.size(), 1);
            auto inputShape = inputs[i]->shape();
            std::copy(inputShape.begin(), inputShape.end(), expanded.end() - inputShape.size());
            auto reshape = operation("reshape");
            reshape.input("x", value).shape("shape", expanded, tag + "_shape");
            auto inputType = mCoreMLBackend->typeConvert(inputs[i]->getType());
            value = reshape.output(expanded, tag, inputType);
            std::vector<int> reps(shape.size());
            for (int j = 0; j < shape.size(); ++j) {
                reps[j] = shape[j] / expanded[j];
            }
            value =
                operation("tile").input("x", value).array("reps", reps, tag + "_reps").output(shape, tag, inputType);
        }
    }
    std::string result;
    if (binaryType == BinaryOpOperation_ATAN2) {
        auto zero = buildScalar(0.0f, "zero");
        auto one = buildScalar(1.0f, "one");
        auto pi = buildScalar(3.141592653589793f, "pi");
        auto halfPi = buildScalar(1.5707963267948966f, "half_pi");
        auto yZero = binary("equal", y, zero, boolean);
        auto safeY = select(yZero, one, y);
        result = unary("atan", binary("real_div", x, safeY, dtype));
        auto xNegative = binary("less", x, zero, boolean);
        auto signedPi = select(xNegative, buildScalar(-3.141592653589793f, "negative_pi"), pi);
        result = select(binary("less", y, zero, boolean), binary("add", result, signedPi, dtype), result);
        auto vertical = select(xNegative, buildScalar(-1.5707963267948966f, "negative_half_pi"), halfPi);
        vertical = select(binary("equal", x, zero, boolean), zero, vertical);
        result = select(yZero, vertical, result);
    } else if (binaryType == BinaryOpOperation_SquaredDifference) {
        auto diff = binary("sub", x, y, dtype);
        result = binary("mul", diff, diff, dtype);
    } else if (binaryType == BinaryOpOperation_DIV && dtype == int32) {
        auto quotient = binary("floor_div", x, y, dtype);
        auto remainder = binary("sub", x, binary("mul", quotient, y, dtype), dtype);
        auto zero = buildScalar(0, "zero");
        auto opposite =
            binary("logical_xor", binary("less", x, zero, boolean), binary("less", y, zero, boolean), boolean);
        auto adjust = binary("logical_and", opposite, binary("not_equal", remainder, zero, boolean), boolean);
        result = select(adjust, binary("add", quotient, buildScalar(1, "one"), dtype), quotient);
    } else if (binaryType == BinaryOpOperation_MOD && dtype == int32) {
        auto quotient = binary("floor_div", x, y, dtype);
        result = binary("sub", x, binary("mul", quotient, y, dtype), dtype);
    } else if (binaryType == BinaryOpOperation_FLOORMOD) {
        auto quotient = binary("floor_div", x, y, dtype);
        result = binary("sub", x, binary("mul", quotient, y, dtype), dtype);
    } else if (binaryType == BinaryOpOperation_LOGICALOR || binaryType == BinaryOpOperation_LOGICALXOR) {
        auto zero = inputs[0]->getType() == halide_type_of<int>() ? buildScalar(0, "zero") : buildScalar(0.0f, "zero");
        x = binary("not_equal", x, zero, boolean);
        y = binary("not_equal", y, zero, boolean);
        result = binary(binaryTypeMap[binaryType], x, y, boolean);
    } else {
        auto optype = binaryTypeMap[binaryType];
        if (optype.empty()) {
            return NOT_SUPPORT;
        }
        bool comparison = binaryType == BinaryOpOperation_GREATER || binaryType == BinaryOpOperation_GREATER_EQUAL ||
                          binaryType == BinaryOpOperation_LESS || binaryType == BinaryOpOperation_LESS_EQUAL ||
                          binaryType == BinaryOpOperation_EQUAL || binaryType == BinaryOpOperation_NOTEQUAL;
        result = binary(optype, x, y, comparison ? boolean : dtype);
    }
    bool comparison = binaryType == BinaryOpOperation_GREATER || binaryType == BinaryOpOperation_GREATER_EQUAL ||
                      binaryType == BinaryOpOperation_LESS || binaryType == BinaryOpOperation_LESS_EQUAL ||
                      binaryType == BinaryOpOperation_EQUAL || binaryType == BinaryOpOperation_NOTEQUAL ||
                      binaryType == BinaryOpOperation_LOGICALOR || binaryType == BinaryOpOperation_LOGICALXOR;
    auto output = operation(comparison ? "cast" : activationType ? "relu" : "identity");
    output.input("x", result);
    if (comparison) {
        output.scalar("dtype", std::string(dtype == int32 ? "int32" : "fp16"), "output_dtype");
    }
    output.output(outputs);
    // TODO: more BinaryType and activation support
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLBinary, OpType_BinaryOp);
REGISTER_COREML_OP_CREATOR(CoreMLBinary, OpType_Eltwise);

} // namespace MNN
