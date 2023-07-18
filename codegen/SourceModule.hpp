//
//  SourceModule.hpp
//  MNN
//
//  Created by MNN on 2022/11/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <vector>
#include <utility>
#include <unordered_map>
#include "core/Command.hpp"
#include "MNN/Tensor.hpp"
#pragma once
namespace MNN {

using InOutTensors = std::pair<std::vector<Tensor*>, std::vector<Tensor*>>;

static inline bool isElemWise(OpType type) {
    return type == OpType_BinaryOp || type == OpType_UnaryOp ||
           type == OpType_Eltwise || type == OpType_ReLU ||
           type == OpType_ReLU6;
}

struct Node {
    std::vector<Node*> pred;
    std::vector<Node*> succ;
    std::vector<Node*> domainateSucc;
    Node* domainatePred;
    Command* cmd;
    int topoIndex;
};

class Target {
public:
    Target(const BackendConfig::PrecisionMode precision) : mPrecision(precision){
        // Nothing
    }
    ~Target() = default;
    virtual std::string codegen(std::vector<std::string>& inputs, const Command* cmd, std::string& inpName) = 0;
    virtual std::string type() = 0;
    virtual std::string macro() = 0;
    virtual std::string number(float val) = 0;
    virtual std::string load(const std::string& base, const std::string& offset, const Command* cmd, std::string& inpName) = 0;
    virtual std::string loadscalar(const std::string& base, std::string& inpName) = 0;
    virtual std::string store(const std::string base, const std::string& offset, const std::string& data) = 0;
    virtual std::string proto(const std::string& name, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, bool hasSingleConvertRaster = false) = 0;
    void setFuseKernelVectorize(bool vector) {
        mVectorize = vector;
    }
    int beginSize() {return mKernelBeginSize; };
protected:
    BackendConfig::PrecisionMode mPrecision;
    bool mVectorize = false;
    int mKernelBeginSize;
};

class SourceModule {
public:
    SourceModule(Target* target) : mTarget(target) {}
    ~SourceModule() = default;
    InOutTensors buildKernel(std::vector<Node*> nodes, int idx);
    std::string codegen();
    std::string kernelName();
    std::string opName();
    int strIndexForKernelNum();
private:
    void down();
    void up();
    std::string getIndent();
    std::string getNameByTensor(Tensor* t, bool read);
private:
    Target* mTarget;
    std::vector<Tensor*> mInputs;
    std::vector<Tensor*> mOutputs;
    std::unordered_map<const Tensor*, std::string> mVarMap;
    std::string mKernelCode, mKernelName, mOpName;
    std::map<std::string, std::pair<std::string, std::string>> mCacheKernel;
    int mIndent = 0;
    int mKernelNumIndex;
};

}
