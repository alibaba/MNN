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
    "#ifdef MNN_SUPPORT_FP16\n"
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "#endif\n"
    "#define OFFSET_CHECK\\\n"
    "\tconst int c = get_global_id(0), w = get_global_id(1), hb = get_global_id(2);\\\n"
    "\tif (c >= global_size_dim0 || w >= global_size_dim1 || hb >= global_size_dim2) { return; }\\\n"
    "\tconst int2 offset = (int2)(mad24(c, global_size_dim1, w), hb);\n"
    "\t__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n";
}
std::string OpenCLTarget::number(float val) {
    return numval(val);
}
std::string OpenCLTarget::codegen(std::vector<std::string>& inputs, const Command* cmd, std::string& inpName) {
    std::stringstream ss; 
    auto op = cmd->op;
    switch (op->type()) {
        case MNN::OpType_BinaryOp:
        {
            auto lhs = inputs[0], rhs = inputs[1];
            auto type = static_cast<MNN::BinaryOpOperation>(op->main_as_BinaryOp()->opType());
            switch (type) {
                case BinaryOpOperation_ADD:
                    ss << inpName << "=(" << lhs << "+" << rhs << ")";
                    break;
                case BinaryOpOperation_SUB:
                    ss << inpName << "=(" << lhs << "-" << rhs << ")";
                    break;
                case BinaryOpOperation_MUL:
                    ss << inpName << "=(" << lhs << "*" << rhs << ")";
                    break;
                case BinaryOpOperation_POW:
                    ss << inpName << "=pow(" << lhs << "," << rhs << ")";
                    break;
                case BinaryOpOperation_DIV:
                    ss << inpName << "=(" << lhs << "/" << rhs << ")";
                    break;
                case BinaryOpOperation_MAXIMUM:
                    ss << inpName << "=fmax(" << lhs << "," << rhs << ")";
                    break;
                case BinaryOpOperation_MINIMUM:
                    ss << inpName << "=fmin(" << lhs << "," << rhs << ")";
                    break;
                case BinaryOpOperation_REALDIV:
                    ss << inpName << "=(" << lhs << "/" << rhs << ")";
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
                    ss << inpName << "=(" << inputs[0] << elemToOp[type] << inputs[1];
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
                    ss << inpName << "=" << fmax(0);
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
                    ss << inpName << "=" << operand << " * " << operand;
                    break;
                case UnaryOpOperation_ERF:
                    ss << inpName << "=CONVERT_FLOAT4(erf(convert_float4(" << operand << ")))";
                    break;
                case UnaryOpOperation_ERFC:
                    ss << inpName << "=CONVERT_FLOAT4(erfc(convert_float4(" << operand << ")))";
                    break;
                case UnaryOpOperation_SQRT:
                    ss << inpName << "=CONVERT_FLOAT4(sqrt(convert_float4(" << operand << ")))";
                    break;
                case UnaryOpOperation_RSQRT:
                    ss << inpName << "=CONVERT_FLOAT4(rsqrt(convert_float4(" << operand << ")))";
                    break;
                case UnaryOpOperation_ABS:
                    ss << inpName << "=CONVERT_FLOAT4(fabs(convert_float4(" << operand << ")))";
                    break;
                case UnaryOpOperation_SIN:
                    ss << inpName << "=CONVERT_FLOAT4(sin(convert_float4(" << operand << ")))";
                    break;
                case UnaryOpOperation_COS:
                    ss << inpName << "=CONVERT_FLOAT4(cos(convert_float4(" << operand << ")))";
                    break;
                case UnaryOpOperation_SIGN:
                    ss << inpName << "=CONVERT_FLOAT4(sign(convert_float4(" << operand << ")))";
                    break;
                case UnaryOpOperation_EXP:
                    ss << inpName << "=CONVERT_FLOAT4(exp(convert_float4(" << operand << ")))";
                    break;
                case UnaryOpOperation_NEG:
                    ss << inpName << "=-(" << operand << ")";
                    break;
                case UnaryOpOperation_TAN:
                    ss << inpName << "=CONVERT_FLOAT4(tan(convert_float4(" << operand << ")))";
                    break;
                case UnaryOpOperation_CEIL:
                    ss << inpName << "=CONVERT_FLOAT4(ceil(convert_float4(" << operand << ")))";
                    break;
                case UnaryOpOperation_LOG1P:
                    ss << inpName << "=CONVERT_FLOAT4(log1p(convert_float4(" << operand << ")))";
                    break;
                case UnaryOpOperation_FLOOR:
                    ss << inpName << "=CONVERT_FLOAT4(floor(convert_float4(" << operand << ")))";
                    break;
                case UnaryOpOperation_ROUND:
                    ss << inpName << "=CONVERT_FLOAT4(round(convert_float4(" << operand << ")))";
                    break;
                case UnaryOpOperation_SIGMOID:
                    ss << inpName << "=CONVERT_FLOAT4(native_recip((float4)1+native_exp(convert_float4(-" << operand << "))))";
                    break;
                case UnaryOpOperation_TANH:
                    ss << inpName << "=CONVERT_FLOAT4(tanh(convert_float4(" << operand << ")))";
                    break;
                case UnaryOpOperation_RECIPROCAL:
                    ss << inpName << "=CONVERT_FLOAT4(native_recip(convert_float4(" << operand << ")))";
                    break;
                case UnaryOpOperation_LOG:
                    ss << inpName << "=CONVERT_FLOAT4(native_log(convert_float4(" << operand << ")+(float4)((float)0.0000001)))";
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
            ss << inpName << "=fmin(fmax(" << operand << "," << numval(minv) << "), " << numval(maxv) << ")";
            break;
        }
        case MNN::OpType_ReLU:
        {
            auto operand = inputs[0];
            auto relu = op->main_as_Relu();
            float slope = relu->slope();
            ss << inpName << "=fmax(" << operand << "," << numval(0) << ")";
            break;
        }
        default:
            break;
    }
    return ss.str();
}
std::string OpenCLTarget::load(const std::string& base, const std::string& offset, const Command* cmd, std::string& inpName) {
    return "FLOAT4 " + inpName + "=RI_F(" + base + ", SAMPLER, " + offset + ")";
}
std::string OpenCLTarget::loadscalar(const std::string& base, std::string& inpName) {
    return "FLOAT4 " + inpName + "=(RI_F(" + base + ", SAMPLER, (int2)(0, 0)).x)";
}
std::string OpenCLTarget::store(const std::string base, const std::string& offset, const std::string& data) {
    return "WI_F(" + base + ", " + offset + ", " + data + ");\n";
}

std::string OpenCLTarget::proto(const std::string& name, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, bool hasSingleConvertRaster) {
    std::stringstream proto;
    std::string begin = "__kernel void ";
    mKernelBeginSize = begin.size();
    proto << begin << "(";
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
