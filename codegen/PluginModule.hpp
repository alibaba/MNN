//
//  Module.hpp
//  MNN
//
//  Created by MNN on 2020/12/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <vector>
#include <utility>
#include "core/Command.hpp"
#include "MNN/Tensor.hpp"

#pragma once

using InOutTensors = std::pair<std::vector<MNN::Tensor*>, std::vector<MNN::Tensor*>>;

static inline bool isElemWise(MNN::OpType type) {
    return type == MNN::OpType_BinaryOp || type == MNN::OpType_UnaryOp ||
           type == MNN::OpType_Eltwise || type == MNN::OpType_ReLU ||
           type == MNN::OpType_ReLU6;
}

struct Node {
    std::vector<Node*> pred;
    std::vector<Node*> succ;
    std::vector<Node*> domainateSucc;
    Node* domainatePred;
    const MNN::Command* cmd;
    int topoIndex;
};

class PluginModule {
public:
    PluginModule() {}
    virtual ~PluginModule() = default;
    virtual InOutTensors addFunction(std::vector<Node*> nodes) = 0;
    virtual const int getFunctionNum() = 0;
    virtual void codegen() = 0;
};

class LLVMTarget;
#ifdef MNN_CODEGEN_CPU
class CPUPluginModule : PluginModule{
public:
    CPUPluginModule();
    CPUPluginModule(std::string name);
    ~CPUPluginModule();
    CPUPluginModule(CPUPluginModule&& m);
    CPUPluginModule& operator=(CPUPluginModule&& m);
    InOutTensors addFunction(std::vector<Node*> nodes) override;
    const int getFunctionNum() override { return functions.size(); }
    void codegen() override;
    void codegen(LLVMTarget* target);
private:
    class CPUPluginFunction;
    std::vector<std::unique_ptr<CPUPluginFunction>> functions;
    std::string name;
};
#endif

#ifdef MNN_CODEGEN_OPENCL
class OpenCLPluginModule : PluginModule {
public:
    OpenCLPluginModule();
    ~OpenCLPluginModule();
    OpenCLPluginModule(OpenCLPluginModule&& m);
    OpenCLPluginModule& operator=(OpenCLPluginModule&& m);
    InOutTensors addFunction(std::vector<Node*> nodes) override;
    const int getFunctionNum() override { return functions.size(); }
    void codegen() override;
private:
    class OpenCLPluginFunction;
    std::vector<std::unique_ptr<OpenCLPluginFunction>> functions;
};
#endif
