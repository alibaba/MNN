//
//  MetalTarget.cpp
//  MNN
//
//  Created by MNN on 2022/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include "MetalTarget.hpp"

namespace MNN {
std::string MetalTarget::type() {
    return "auto ";
}
std::string MetalTarget::macro() {
    return
    "using namespace metal;\n"
    "struct unary_shape {\n"
    " int width;\n"
    " int height;\n"
    " int size;\n"
    "};\n"
    "#define OFFSET_CHECK\\\n"
    "\tif (gid.x >= (uint)s.width) { return; }\\\n"
    "\tint offset=gid.z*s.size+gid.y*s.width+gid.x;\n";
}
std::string MetalTarget::number(float val) {
    return numval(val);
}
std::string MetalTarget::codegen(std::vector<std::string>& inputs, const Op* op) {
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
                    ss << "(" << operand << " * " << operand << ")";
                    break;
                case UnaryOpOperation_SQRT:
                    ss << "sqrt(" << operand << ")";
                    break;
                case UnaryOpOperation_RSQRT:
                    ss << "rsqrt(" << operand << ")";
                    break;
                case UnaryOpOperation_ABS:
                    ss << "abs(" << operand << ")";
                    break;
                case UnaryOpOperation_SIN:
                    ss << "sin(" << operand << ")";
                    break;
                case UnaryOpOperation_COS:
                    ss << "cos(" << operand << ")";
                    break;
                case UnaryOpOperation_SIGN:
                    ss << "sign(" << operand << ")";
                    break;
                case UnaryOpOperation_EXP:
                    ss << "exp(" << operand << ")";
                    break;
                case UnaryOpOperation_NEG:
                    ss << "-(" << operand << ")";
                    break;
                case UnaryOpOperation_TAN:
                    ss << "tan(" << operand << ")";
                    break;
                case UnaryOpOperation_CEIL:
                    ss << "ceil(" << operand << ")";
                    break;
                case UnaryOpOperation_LOG1P:
                    ss << "log(1.f + " << operand << ")";
                    break;
                case UnaryOpOperation_FLOOR:
                    ss << "floor(" << operand << ")";
                    break;
                case UnaryOpOperation_ROUND:
                    ss << "round(" << operand << ")";
                    break;
                case UnaryOpOperation_SIGMOID:
                    ss << "(1.f / (1.f + exp(-" << operand << ")))";
                    break;
                case UnaryOpOperation_TANH:
                    ss << "tanh(" << operand << ")";
                    break;
                case UnaryOpOperation_RECIPROCAL:
                    ss << "(1.0 / " << operand << ")";
                    break;
                case UnaryOpOperation_LOG:
                    ss << "log(" << operand << ")";
                    break;
                default:
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
            ss << "clamp(" << operand << ", " << numval(minv) << ", " << numval(maxv) << ")";
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
std::string MetalTarget::load(const std::string& base, const std::string& offset) {
    return "(M4)(" + base + "[" + offset + "])";
}
std::string MetalTarget::loadscalar(const std::string& base) {
    return "(M4)(" + base + "[0])";
}
std::string MetalTarget::store(const std::string base, const std::string& offset, const std::string& data) {
    return base + "[" + offset + "] = " + data + ";\n";
}

std::string MetalTarget::proto(const std::string& name, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs) {
    std::stringstream proto;
    int buffer_idx = 0;
    proto << "kernel void " << name << "(";
    for (auto& input : inputs) {
        proto << "const device M4* " << input << " [[buffer(" << buffer_idx++ << ")]], ";
    }
    for (auto& output : outputs) {
        proto << "device M4* " << output << " [[buffer(" << buffer_idx++ << ")]], ";
    }
    proto << "device unary_shape& s [[buffer(" << buffer_idx++ << ")]], ";
    proto << "uint3 gid [[thread_position_in_grid]])";
    return proto.str();
}
} // MNN
