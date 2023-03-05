//
//  SourceModule.cpp
//  MNN
//
//  Created by MNN on 2020/12/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include "AST.hpp"

using namespace AST;
using namespace MNN;

std::vector<std::vector<Node*>> spliteNodes(std::vector<Node*>& nodes) {
    std::vector<std::vector<Node*>> res;
    res.push_back(std::vector<Node*>());
    for (auto node : nodes) {
        if (isElemWise(node->cmd->op->type())) {
            res.back().push_back(node);
        } else {
            if (res.back().empty()) {
                res.back().push_back(node);
            } else {
                res.push_back({node});
            }
            res.push_back(std::vector<Node*>());
        }
    }
    if (res.back().empty()) {
        res.pop_back();
    }
    return res;
}

class SourceModule::CPUPluginFunction {
public:
    CPUPluginFunction(std::vector<Node*>& nodes, int idx) : nodes(nodes) {
        sort(nodes.begin(), nodes.end(), [](Node* x, Node* y) { return x->topoIndex < y->topoIndex; });
        std::unique_ptr<ListExpr> list(new ListExpr);
        auto subNodes = spliteNodes(nodes);
        for (auto subNode : subNodes) {
            if (subNode.size() > 1 || subNode.back()->cmd->op->type() != OpType_Raster) {
                auto loop = addElemwiseLoop(subNode);
                list->push_back(std::move(loop));
            } else {
                auto raster = addRaster(subNode);
                list->push_back(std::move(raster));
            }
        }
        auto proto = std::make_unique<PrototypeAST>("kernel_" + std::to_string(idx), inputs.size(), outputs.size());
        function = std::make_unique<FunctionAST>(std::move(proto), std::move(list));
    }
    std::string codegen(SourceTarget* target) {
        return function->codegen(target);
    }
    std::vector<Tensor*> getInputs() { return inputs; }
    std::vector<Tensor*> getOutputs() { return outputs; }
private:
    std::unique_ptr<Expr> addElemwiseLoop(std::vector<Node*>& nodes) {
        std::map<const Tensor*, int> varShape;
        std::unordered_map<Tensor*, std::unique_ptr<Expr>> outMap;
        for (auto& node : nodes) {
            auto cmd = node->cmd;
            std::vector<std::unique_ptr<Expr>> in(cmd->inputs.size());
            for (int i = 0; i < cmd->inputs.size(); i++) {
                if (outMap.find(cmd->inputs[i]) == outMap.end()) {
                    auto inputExpr = getExprByTensor(cmd->inputs[i], true);
                    int size = cmd->inputs[i]->elementSize();
                    varShape[cmd->inputs[i]] = size;
                    if (size > 1) {
                        in[i] = std::make_unique<SubscriptExpr>(std::move(inputExpr), "i");
                    } else {
                        in[i] = std::make_unique<SubscriptExpr>(std::move(inputExpr), 0);
                    }
                } else {
                    in[i] = std::move(outMap[cmd->inputs[i]]);
                    outMap.erase(cmd->inputs[i]);
                }
            }
            switch (cmd->op->type()) {
                case MNN::OpType_BinaryOp:
                {
                    auto type = static_cast<MNN::BinaryOpOperation>(cmd->op->main_as_BinaryOp()->opType());
                    outMap[cmd->outputs[0]] = std::make_unique<BinaryExpr>(type, std::move(in[0]), std::move(in[1]));
                    break;
                }
                case MNN::OpType_Eltwise:
                {
                    std::map<EltwiseType, MNN::BinaryOpOperation> elemToBinary = {
                        {EltwiseType_PROD, BinaryOpOperation_MUL},
                        {EltwiseType_SUM, BinaryOpOperation_ADD},
                        {EltwiseType_MAXIMUM, BinaryOpOperation_MAXIMUM},
                        {EltwiseType_SUB, BinaryOpOperation_SUB}
                    };
                    auto type = elemToBinary[cmd->op->main_as_Eltwise()->type()];
                    auto tmp = std::make_unique<BinaryExpr>(type, std::move(in[0]), std::move(in[1]));
                    for (int i = 2; i < cmd->inputs.size(); i++) {
                        tmp = std::make_unique<BinaryExpr>(type, std::move(tmp), std::move(in[i]));
                    }
                    outMap[cmd->outputs[0]] = std::move(tmp);
                    break;
                }
                case MNN::OpType_UnaryOp:
                {
                    auto unary = cmd->op->main_as_UnaryOp();
                    auto type = unary->opType();
                    outMap[cmd->outputs[0]] = std::make_unique<UnaryExpr>(type, std::move(in[0]));
                    break;
                }
                case MNN::OpType_ReLU6:
                {
                    auto relu6 = cmd->op->main_as_Relu6();
                    float minv = relu6->minValue();
                    float maxv = relu6->maxValue();
                    outMap[cmd->outputs[0]] = std::make_unique<ReluExpr>(minv, maxv, std::move(in[0]));
                    break;
                }
                case MNN::OpType_ReLU:
                {
                    auto relu = cmd->op->main_as_Relu();
                    float slope = relu->slope();
                    outMap[cmd->outputs[0]] = std::make_unique<ReluExpr>(slope, 0, std::move(in[0]));
                    break;
                }
                default:
                    break;
            }
        }
        std::unique_ptr<Expr> content;
        for (auto& iter : outMap) {
            auto outputExpr = getExprByTensor(iter.first, false);
            auto output = std::make_unique<SubscriptExpr>(std::move(outputExpr), "i");
            varShape[iter.first] = iter.first->elementSize();
            content = std::make_unique<AssignExpr>(std::move(output), std::move(iter.second));
        }
        int size = -1;
        for (auto& iter : varShape) {
            if (iter.second > 1) {
                if (size > 1 && iter.second != size) {
                    MNN_ERROR("size not equal!\n");
                    exit(0);
                } else {
                    size = iter.second;
                }
            }
        }
        auto start = std::make_unique<NumberExpr>(0);
        auto end = std::make_unique<NumberExpr>(size);
        auto step = std::make_unique<NumberExpr>(1);
        auto loop = std::make_unique<ForExpr>("i", std::move(start), std::move(end), std::move(step), std::move(content));
        return loop;
    }

    std::unique_ptr<Expr> addRaster(std::vector<Node*>& nodes) {
        auto node = nodes.back();
        auto input = node->cmd->inputs[0];
        auto output = node->cmd->outputs[0];
        auto des = TensorUtils::getDescribe(input);
        std::string foots[3] = { "i", "j", "k" };
        auto getOffset = [&foots](int strides[], int offset) {
            std::unique_ptr<Expr> steps[3];
            for (int i = 0; i < 3; i++) {
                auto stride = std::make_unique<NumberExpr>(strides[i]);
                auto foot = std::make_unique<VariableExpr>(foots[i]);
                steps[i] = std::make_unique<BinaryExpr>(MNN::BinaryOpOperation_MUL, std::move(foot), std::move(stride));
            }
            auto res = std::make_unique<BinaryExpr>(MNN::BinaryOpOperation_ADD, std::move(steps[1]), std::move(steps[2]));
            res = std::make_unique<BinaryExpr>(MNN::BinaryOpOperation_ADD, std::move(steps[0]), std::move(res));
            return std::make_unique<BinaryExpr>(MNN::BinaryOpOperation_ADD, std::move(res), std::make_unique<NumberExpr>(offset));
        };
        std::unique_ptr<ListExpr> list(new ListExpr);
        for (auto& reg : des->regions) {
            auto inputExpr = getExprByTensor(reg.origin, true);
            auto outputExpr = getExprByTensor(output, false);
            auto srcPtr = std::make_unique<SubscriptExpr>(std::move(inputExpr), getOffset(reg.src.stride, reg.src.offset));
            auto dstPtr = std::make_unique<SubscriptExpr>(std::move(outputExpr), getOffset(reg.dst.stride, reg.dst.offset));
            std::unique_ptr<Expr> content = std::make_unique<AssignExpr>(std::move(dstPtr), std::move(srcPtr));
            for (int i = 2; i >= 0; i--) {
                auto start = std::make_unique<NumberExpr>(0);
                auto end = std::make_unique<NumberExpr>(reg.size[i]);
                auto step = std::make_unique<NumberExpr>(1);
                content = std::make_unique<ForExpr>(foots[i], std::move(start), std::move(end), std::move(step), std::move(content));
            }
            list->push_back(std::move(content));
        }
        return list;
    }
    std::unique_ptr<Expr> getExprByTensor(Tensor* t, bool read) {
        if (inputMap.find(t) != inputMap.end()) {
            return std::make_unique<SubscriptExpr>("inputs", inputMap[t]);
        }
        if (outputMap.find(t) != outputMap.end()) {
            return std::make_unique<SubscriptExpr>("outputs", outputMap[t]);
        }
        if (read) {
            int idx = inputs.size();
            inputs.push_back(t);
            inputMap[t] = idx;
            return std::make_unique<SubscriptExpr>("inputs", idx);
        } else {
            int idx = outputs.size();
            outputs.push_back(t);
            outputMap[t] = idx;
            return std::make_unique<SubscriptExpr>("outputs", idx);
        }
    }
private:
    std::vector<Node*> nodes;
    std::vector<Tensor*> inputs;
    std::vector<Tensor*> outputs;
    std::unordered_map<const Tensor*, int> inputMap;
    std::unordered_map<const Tensor*, int> outputMap;
    std::unique_ptr<FunctionAST> function;
};

void CPUPluginModule::codegen(LLVMTarget* target) {
    for (int i = 0; i < getFunctionNum(); i++) {
        functions[i]->codegen(target);
    }
}

void CPUPluginModule::codegen() {
    std::ofstream headerFile("./kernel.h");
    std::ofstream sourceFile("./kernel.c");
    if (!headerFile.is_open()) {
        return;
    }
    headerFile << "extern \"C\" {\n";
#ifdef MNN_CODEGEN_LLVM
    std::unique_ptr<LLVMTarget> llvm(new LLVMTarget(name));
#endif
#ifdef MNN_CODEGEN_C
    sourceFile << "#include \"math.h\"\n";
    std::unique_ptr<SourceTarget> source(new CTarget(name));
#endif
    for (int i = 0; i < getFunctionNum(); i++) {
        headerFile << "void kernel_" + std::to_string(i) + "(float**, float**);\n";
#ifdef MNN_CODEGEN_C
        sourceFile << functions[i]->codegen(source.get());
#endif
#ifdef MNN_CODEGEN_LLVM
        functions[i]->codegen(llvm.get());
#endif
    }
    headerFile << "}\n";
    headerFile << "void (*kernels[])(float**, float**)  = {\n";
    for (int i = 0; i < getFunctionNum(); i++) {
        headerFile << "&kernel_" + std::to_string(i) + ",\n";
    }
    headerFile << "};\n";
#ifdef MNN_CODEGEN_LLVM
    // write to bc file
    std::error_code EC;
    llvm::raw_fd_ostream OS("kernel.bc", EC, sys::fs::F_None);
    WriteBitcodeToFile(*llvm->getModule(), OS);
    OS.flush();
#endif
}

InOutTensors CPUPluginModule::addFunction(std::vector<Node*> nodes) {
    std::unique_ptr<CPUPluginFunction> func(new CPUPluginFunction(nodes, getFunctionNum()));
    auto res = std::make_pair<std::vector<MNN::Tensor*>, std::vector<MNN::Tensor*>>(func->getInputs(), func->getOutputs());
    functions.emplace_back(std::move(func));
    return res;
}

CPUPluginModule::CPUPluginModule() {}
CPUPluginModule::CPUPluginModule(std::string name) : name(name) {}
CPUPluginModule::~CPUPluginModule() = default;
CPUPluginModule::CPUPluginModule(CPUPluginModule&& m) = default;
CPUPluginModule& CPUPluginModule::operator=(CPUPluginModule&& m) = default;
