//
//  OpenCLTarget.cpp
//  MNN
//
//  Created by MNN on 2022/11/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include "OpenCLTarget.hpp"

namespace MNN {
std::string OpenCLTarget::type() {
    return "FLOAT4 ";
}
std::string OpenCLTarget::macro() {
    return
    "#define OFFSET_CHECK\\\n"
    "\tconst int c = get_global_id(0), w = get_global_id(1), hb = get_global_id(2);\\\n"
    "\tif (c >= global_size_dim0 || w >= global_size_dim1 || hb >= global_size_dim2) { return; }\\\n"
    "\tconst int2 offset = (int2)(mad24(c, global_size_dim1, w), hb);\n"
    "\t__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n";
}
std::string OpenCLTarget::number(float val) {
    return numval(val);
}
std::string OpenCLTarget::codegen(std::vector<std::string>& inputs, const Op* op) {
    std::stringstream ss; 
    switch (op->type()) {
        case MNN::OpType_BinaryOp:
        {
            auto lhs = inputs[0], rhs = inputs[1];
            auto type = static_cast<MNN::BinaryOpOperation>(op->main_as_BinaryOp()->opType());
            switch (type) {
                case BinaryOpOperation_ADD:
                    ss << "(" << lhs << "+" << rhs << ")";
                    break;
                case BinaryOpOperation_SUB:
                    ss << "(" << lhs << "-" << rhs << ")";
                    break;
                case BinaryOpOperation_MUL:
                    ss << "(" << lhs << "*" << rhs << ")";
                    break;
                case BinaryOpOperation_POW:
                    ss << "pow(" << lhs << "," << rhs << ")";
                    break;
                case BinaryOpOperation_DIV:
                    ss << "(" << lhs << "/" << rhs << ")";
                    break;
                case BinaryOpOperation_MAXIMUM:
                    ss << "fmax(" << lhs << "," << rhs << ")";
                    break;
                case BinaryOpOperation_MINIMUM:
                    ss << "fmin(" << lhs << "," << rhs << ")";
                    break;
                case BinaryOpOperation_REALDIV:
                    ss << "(" << lhs << "/" << rhs << ")";
                    break;
                default:
                    break;
            }
            break;
        }
        case MNN::OpType_Eltwise:
        {
            auto type = op->main_as_Eltwise()->type();
            switch (type) {
                case EltwiseType_SUM:
                case EltwiseType_SUB:
                case EltwiseType_PROD:
                {
                    std::unordered_map<int, std::string> elemToOp = {
                        {EltwiseType_PROD, "*"}, {EltwiseType_SUM, "+"}, {EltwiseType_SUB, "-"}
                    };
                    ss << "(" << inputs[0] << elemToOp[type] << inputs[1];
                    for (int i = 2; i < inputs.size(); i++) {
                        ss << elemToOp[type] << inputs[i];
                    }
                    ss << ")";
                    break;
                }
                case EltwiseType_MAXIMUM:
                {
                    std::function<std::string(int)> fmax = [&inputs, &fmax](int d) {
                        if (d == inputs.size() - 1) {
                            return inputs[d];
                        }
                        return "fmax(" + inputs[d] + ", " + fmax(d+1) + ")";
                    };
                    ss << fmax(0);
                    break;
                }
                default:
                    break;
            }
            break;
        }
        case MNN::OpType_UnaryOp:
        {
            auto unary = op->main_as_UnaryOp();
            auto type = unary->opType();
            auto operand = inputs[0];
            switch (type) {
                case UnaryOpOperation_SQUARE:
                    ss << operand << " * " << operand;
                    break;
                case UnaryOpOperation_ERF:
                    ss << "erf(convert_float4(" << operand << "))";
                    break;
                case UnaryOpOperation_ERFC:
                    ss << "erfc(convert_float4(" << operand << "))";
                    break;
                case UnaryOpOperation_SQRT:
                    ss << "sqrt(convert_float4(" << operand << "))";
                    break;
                case UnaryOpOperation_RSQRT:
                    ss << "rsqrt(convert_float4(" << operand << "))";
                    break;
                case UnaryOpOperation_ABS:
                    ss << "fabs(convert_float4(" << operand << "))";
                    break;
                case UnaryOpOperation_SIN:
                    ss << "sin(convert_float4(" << operand << "))";
                    break;
                case UnaryOpOperation_COS:
                    ss << "cos(convert_float4(" << operand << "))";
                    break;
                case UnaryOpOperation_SIGN:
                    ss << "sign(convert_float4(" << operand << "))";
                    break;
                case UnaryOpOperation_EXP:
                    ss << "exp(convert_float4(" << operand << "))";
                    break;
                case UnaryOpOperation_NEG:
                    ss << "-(" << operand << ")";
                    break;
                case UnaryOpOperation_TAN:
                    ss << "tan(convert_float4(" << operand << "))";
                    break;
                case UnaryOpOperation_CEIL:
                    ss << "ceil(convert_float4(" << operand << "))";
                    break;
                case UnaryOpOperation_LOG1P:
                    ss << "log1p(convert_float4(" << operand << "))";
                    break;
                case UnaryOpOperation_FLOOR:
                    ss << "floor(convert_float4(" << operand << "))";
                    break;
                case UnaryOpOperation_ROUND:
                    ss << "round(convert_float4(" << operand << "))";
                    break;
                case UnaryOpOperation_SIGMOID:
                    ss << "native_recip((float4)1+native_exp(convert_float4(-" << operand << ")))";
                    break;
                case UnaryOpOperation_TANH:
                    ss << "tanh(convert_float4(" << operand << "))";
                    break;
                case UnaryOpOperation_RECIPROCAL:
                    ss << "native_recip(convert_float4(" << operand << "))";
                    break;
                case UnaryOpOperation_LOG:
                    ss << "native_log(convert_float4(" << operand << "+(float4)((float)0.0000001)))";
                    break;
                default:
                    MNN_ASSERT(false);
                    break;
            }
            break;
        }
        case MNN::OpType_ReLU6:
        {
            auto operand = inputs[0];
            auto relu6 = op->main_as_Relu6();
            float minv = relu6->minValue();
            float maxv = relu6->maxValue();
            ss << "fmin(fmax(" << operand << "," << numval(minv) << "), " << numval(maxv) << ")";
            break;
        }
        case MNN::OpType_ReLU:
        {
            auto operand = inputs[0];
            auto relu = op->main_as_Relu();
            float slope = relu->slope();
            ss << "fmax(" << operand << "," << numval(0) << ")";
            break;
        }
        default:
            break;
    }
    return ss.str();
}
std::string OpenCLTarget::load(const std::string& base, const std::string& offset) {
    return "read_imagef(" + base + ", SAMPLER, " + offset + ")";
}
std::string OpenCLTarget::loadscalar(const std::string& base) {
    return "((float4)read_imagef(" + base + ", SAMPLER, (int2)(0, 0)).x)";
}
std::string OpenCLTarget::store(const std::string base, const std::string& offset, const std::string& data) {
    return "write_imagef(" + base + ", " + offset + ", " + data + ");\n";
}

std::string OpenCLTarget::proto(const std::string& name, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs) {
    std::stringstream proto;
    proto << "__kernel void " << name << "(";
    for (auto& input : inputs) {
        proto << "__read_only image2d_t " << input << ", ";
    }
    for (auto& output : outputs) {
        proto << "__write_only image2d_t " << output << ", ";
    }
    proto << "__private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2)";
    return proto.str();
}
} // MNN
