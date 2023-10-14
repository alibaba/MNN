//
//  SourceModule.cpp
//  MNN
//
//  Created by MNN on 2022/11/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <utility>
#include <unordered_map>

#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include "SourceModule.hpp"
namespace MNN {

class VarScope {
public:
    VarScope() {}
    bool hasVar(const Tensor* t) {
        return mVarMap.find(t) != mVarMap.end();
    }
    std::string addVar(const Tensor* t) {
        std::string name = "var_" + std::to_string(mVarIdx++);
        mVarMap.insert(std::make_pair(t, name));
        return name;
    }
    std::string addInput(const Tensor* t) {
        std::string name = "input_" + std::to_string(mInputIdx++);
        mInputs.push_back(const_cast<Tensor*>(t));
        mIOMap.insert(std::make_pair(t, name));
        return name;
    }
    std::string addOutput(const Tensor* t) {
        std::string name = "output_" + std::to_string(mOutputIdx++);
        mIOMap.insert(std::make_pair(t, name));
        return name;
    }
    std::string getVar(const Tensor* t) {
        return mVarMap[t];
    }
    std::string getIO(const Tensor* t) {
        return mIOMap[t];
    }
    void setUse(const Tensor* t) {
        mUsed.insert(t);
    }
    void computeOutput() {
        mOutputs.clear();
        for (auto& iter : mVarMap) {
            if (mUsed.find(iter.first) == mUsed.end()) {
                mOutputs.push_back(const_cast<Tensor*>(iter.first));
            }
        }
    }
    InOutTensors getIOTensors() {
        return { mInputs, mOutputs };
    }
private:
    int mInputIdx = 0, mOutputIdx = 0, mVarIdx = 0;
    std::unordered_map<const Tensor*, std::string> mVarMap;
    std::unordered_map<const Tensor*, std::string> mIOMap;
    std::set<const Tensor*> mUsed;
    std::vector<Tensor*> mInputs, mOutputs;
};

static std::string opStr(const Op* op) {
    std::stringstream ss;
    ss << EnumNameOpType(op->type()) << "[";
    switch (op->type())
    {
        case OpType_BinaryOp:
            ss << EnumNameBinaryOpOperation(static_cast<MNN::BinaryOpOperation>(op->main_as_BinaryOp()->opType()));
            break;
        case OpType_UnaryOp:
            ss << EnumNameUnaryOpOperation(static_cast<MNN::UnaryOpOperation>(op->main_as_UnaryOp()->opType()));
            break;
        case OpType_Eltwise:
            ss << EnumNameEltwiseType(static_cast<MNN::EltwiseType>(op->main_as_Eltwise()->type()));
            break;
        default:
            break;
    }
    ss << "]_";
    return ss.str();
}

InOutTensors SourceModule::buildKernel(std::vector<Node*> nodes, int idx)  {
    VarScope scope;
    std::sort(nodes.begin(), nodes.end(), [](Node* x, Node* y) { return x->topoIndex < y->topoIndex; });
    for (auto& node : nodes) {
        mOpName.append(opStr(node->cmd->op));
    }

    mKernelName = "kernel_" + std::to_string(idx);
    // 0. gen kernel macro
    std::string kernelMacro = mTarget->macro();
    // 1. gen kernel body
    std::stringstream kernelBody;
    kernelBody << "{\n";
    down();
    kernelBody << getIndent() << "OFFSET_CHECK;\n";
    std::string offset = "offset";
    std::unordered_map<MNN::Tensor*, std::string> cacheMap;
    bool singleConvertRaster = false;
    for (auto& node : nodes) {
        auto cmd = node->cmd;
        std::vector<std::string> inputs(cmd->inputs.size());
        for (int i = 0; i < cmd->inputs.size(); i++) {
            if(cmd->op->type() == MNN::OpType_Raster) {
                singleConvertRaster = true;
            }
            auto t = cmd->inputs[i];
            if (scope.hasVar(t)) {
                inputs[i] = scope.getVar(t);
            } else {
                inputs[i] = scope.addVar(t);
                std::string code;
                if ((cmd->inputs[i]->shape().empty() || cmd->inputs[i]->elementSize() == 1) &&
		            TensorUtils::getDescribe(cmd->inputs[i])->usage == Tensor::InsideDescribe::CONSTANT) {
                    float val = cmd->inputs[i]->host<float>()[0];
                    code = mTarget->type() + inputs[i] + "=" + mTarget->number(val);
                } else {
                    if (cmd->inputs[i]->elementSize() == 1) {
                        code = mTarget->loadscalar(scope.addInput(t), inputs[i]);
                    } else {
                        code = mTarget->load(scope.addInput(t), offset, cmd, inputs[i]);
                    }
                }
                kernelBody << getIndent() << code << ";\n";
            }
            scope.setUse(t);
        }
        auto tmpVar = scope.addVar(cmd->outputs[0]);
        kernelBody << getIndent() << mTarget->type() << tmpVar << ";\n";
        std::string computeCode = mTarget->codegen(inputs, cmd, tmpVar);
        kernelBody << getIndent() << computeCode << ";\n";
    }
    scope.computeOutput();
    auto res = scope.getIOTensors();
    for (auto t : res.second) {
        kernelBody << getIndent() << mTarget->store(scope.addOutput(t), offset, scope.getVar(t));
    }
    up();
    kernelBody << "}\n";
    mKernelCode.append(mTarget->macro());
    std::vector<std::string> inputArgs, outputArgs;

    for (auto& input : res.first) {
        inputArgs.push_back(scope.getIO(input));
    }
    for (auto& output : res.second) {
        outputArgs.push_back(scope.getIO(output));
    }
    mKernelCode.append(mTarget->proto(kernelName(), inputArgs, outputArgs, singleConvertRaster));
    mKernelCode.append(kernelBody.str());
    mKernelNumIndex = mTarget->macro().size() + mTarget->beginSize();
    return res;
}

std::string SourceModule::codegen() { return mKernelCode; }

std::string SourceModule::kernelName() { return mKernelName; }

std::string SourceModule::opName() { return mOpName; }

int SourceModule::strIndexForKernelNum() { return mKernelNumIndex; }

void SourceModule::down() { mIndent++; }

void SourceModule::up() { mIndent--; }

std::string SourceModule::getIndent() {
    return std::string(mIndent*4, ' ');
}
} // MNN
