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

using namespace MNN;

class OpenCLTarget {
public:
    OpenCLTarget(std::vector<Node*>& nodes, int idx) : nodes(nodes) {
        sort(nodes.begin(), nodes.end(), [](Node* x, Node* y) { return x->topoIndex < y->topoIndex; });
        // 1. gen kernel body
        std::stringstream kernelBody;
        kernelBody << "{\n";
        down();
        /*
        kernelBody << getIndent() << "const int x = get_global_id(0), y = get_global_id(1);\n";
        kernelBody << getIndent() << "if (x >= h || y >= w) { return; }\n";
        kernelBody << getIndent() << "const int2 pos = (int2)(x, y);\n";
        */
        kernelBody << getIndent() << "GET_CHECK\n";
        // now just deal elemwise
        kernelBody << addElemwiseOp(nodes);
        //addElemwiseOp(nodes);
        //kernelBody << "\twrite_imagef(output_0, pos, read_imagef(input_0, SAMPLER, pos));\n";
        // kernelBody << "write_imagef(output_0, pos, (float4)(1.0, 1.0, 1.0, 1.0));\n";
        up();
        kernelBody << "}\n";
        // 2. gen kernel prototype
        std::stringstream kernelProto;
        // a) func name
        kernelProto << "__kernel void kernel_" << idx << "(";
        // b) args
        for (auto& input : inputs) {
            kernelProto << "__read_only image2d_t " << varMap[input] << ", ";
        }
        for (auto& output : outputs) {
            kernelProto << "__write_only image2d_t " << varMap[output] << ", ";
        }
        // c) dims info
        kernelProto << "__private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2)";
        // 3. append to kernel
        kernelCode.append(kernelProto.str());
        kernelCode.append(kernelBody.str());
    }
    std::string codegen() { return kernelCode; }
    std::vector<Tensor*> getInputs() { return inputs; }
    std::vector<Tensor*> getOutputs() { return outputs; }
private:
    std::string addElemwiseOp(std::vector<Node*>& nodes) {
        std::string pos = "pos";
        std::unordered_map<Tensor*, std::string> cacheMap;
        for (auto& node : nodes) {
            std::stringstream ss;
            auto cmd = node->cmd;
            std::vector<std::string> inputs(cmd->inputs.size());
            for (int i = 0; i < cmd->inputs.size(); i++) {
                if (cacheMap.find(cmd->inputs[i]) == cacheMap.end()) {
                    if (cmd->inputs[i]->shape().empty() && TensorUtils::getDescribe(cmd->inputs[i])->usage == Tensor::InsideDescribe::CONSTANT) {
                        float val = cmd->inputs[i]->host<float>()[0];
                        std::stringstream ssval;
                        ssval << "((float4)(" << val << "))";
                        inputs[i] = ssval.str();
                    } else {
                        inputs[i] = readPixel(getNameByTensor(cmd->inputs[i], true), pos);
                    }
                } else {
                    inputs[i] = cacheMap[cmd->inputs[i]];
                    cacheMap.erase(cmd->inputs[i]);
                }
            }
            switch (cmd->op->type()) {
                case MNN::OpType_BinaryOp:
                {
                    auto lhs = inputs[0], rhs = inputs[1];
                    auto type = static_cast<MNN::BinaryOpOperation>(cmd->op->main_as_BinaryOp()->opType());
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
                    auto type = cmd->op->main_as_Eltwise()->type();
                    switch (type) {
                        case EltwiseType_SUM:
                        case EltwiseType_SUB:
                        case EltwiseType_PROD:
                        {
                            std::unordered_map<EltwiseType, std::string> elemToOp = {
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
                    auto unary = cmd->op->main_as_UnaryOp();
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
                            break;
                    }
                    break;
                }
                case MNN::OpType_ReLU6:
                {
                    auto operand = inputs[0];
                    auto relu6 = cmd->op->main_as_Relu6();
                    float minv = relu6->minValue();
                    float maxv = relu6->maxValue();
                    ss << "fmin(fmax(" << operand << "," << getNumVal(minv) << "), " << getNumVal(maxv) << ")";
                    break;
                }
                case MNN::OpType_ReLU:
                {
                    auto operand = inputs[0];
                    auto relu = cmd->op->main_as_Relu();
                    float slope = relu->slope();
                    ss << "fmax(" << operand << "," << getNumVal(0) << ")";
                    break;
                }
                default:
                    break;
            }
            cacheMap[cmd->outputs[0]] = ss.str();
        }
        std::stringstream ss;
        for (auto& iter : cacheMap) {
            auto output = getNameByTensor(iter.first, false);
            ss << writePixel(output, pos, iter.second);
        }
        return ss.str();
    }
    template <typename T>
    std::string getNumVal(T t) {
        return "(float4)((float)" + std::to_string(t) + ")";
    }
    std::string readPixel(std::string img, std::string pos) {
        return "read_imagef(" + img + ", SAMPLER, " + pos + ")";
    }
    std::string writePixel(std::string img, std::string pos, std::string data) {
        return getIndent() + "write_imagef(" + img + ", " + pos + ", " + data + ");\n";
    }
    void down() { indent++; }
    void up() { indent--; }
    std::string getIndent() {
        return std::string(indent*4, ' ');
    }
    std::string getNameByTensor(Tensor* t, bool read) {
        if (varMap.find(t) != varMap.end()) {
            return varMap[t];
        }
        if (read) {
            int idx = inputs.size();
            inputs.push_back(t);
            varMap[t] = "input_" + std::to_string(idx);
            return varMap[t];
        } else {
            int idx = outputs.size();
            outputs.push_back(t);
            varMap[t] = "output_" + std::to_string(idx);;
            return varMap[t];
        }
    }
private:
    std::vector<Node*> nodes;
    std::vector<Tensor*> inputs;
    std::vector<Tensor*> outputs;
    std::unordered_map<const Tensor*, std::string> varMap;
    std::string kernelCode;
    int indent = 0;
};

std::string OpenCLPluginModule::codegen() {
    std::stringstream sourceCode;
    sourceCode << "#define GET_CHECK\\\n\
        const int c = get_global_id(0), w = get_global_id(1), hb = get_global_id(2);\\\n\
        if (c >= global_size_dim0 || w >= global_size_dim1 || hb >= global_size_dim2) { return; }\\\n\
        const int2 pos = (int2)(mad24(c, global_size_dim1, w), hb);\n";
    sourceCode << "__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n";
    for (int i = 0; i < getFunctionNum(); i++) {
        sourceCode << functions[i]->codegen();
    }
    return sourceCode.str();
}

InOutTensors OpenCLPluginModule::addFunction(std::vector<Node*> nodes) {
    std::unique_ptr<OpenCLPluginFunction> func(new OpenCLPluginFunction(nodes, getFunctionNum()));
    auto res = std::make_pair<std::vector<MNN::Tensor*>, std::vector<MNN::Tensor*>>(func->getInputs(), func->getOutputs());
    functions.emplace_back(std::move(func));
    return res;
}