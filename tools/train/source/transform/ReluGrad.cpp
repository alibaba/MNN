//
//  ReluGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ReluGrad.hpp"
#include "Macro.h"
using namespace std;
using namespace MNN;

class ReluGrad : public OpGrad {
public:
    ReluGrad() {
        mType = SEMI_LINEAR;
    }
    virtual OpConverter::Result onGrad(const MNN::NetT* net, const MNN::OpT* forwardOp,
                                       std::map<int, std::vector<int>>& backwardTensors,
                                       const std::vector<int>& gradTensors) {
        OpConverter::Result result;
        result.newTensorOffset = net->tensorName.size();
        auto outputIndex       = forwardOp->outputIndexes[0];
        auto outputDiff        = backwardTensors.find(outputIndex)->second[0];

        unique_ptr<OpT> newOp(new OpT);
        newOp->name          = forwardOp->name + "_Grad";
        newOp->inputIndexes  = {forwardOp->inputIndexes[0], outputDiff};
        newOp->outputIndexes = {gradTensors[0]};
        newOp->type          = OpType_ReluGrad;
        newOp->main.type     = OpParameter_Relu;
        newOp->main.value    = new ReluT(*forwardOp->main.AsRelu());
        result.opLists.emplace_back(std::move(newOp));

        return result;
    }
};
class Relu6Grad : public OpGrad {
public:
    Relu6Grad() {
        mType = SEMI_LINEAR;
    }

    virtual OpConverter::Result onGrad(const MNN::NetT* net, const MNN::OpT* forwardOp,
                                       std::map<int, std::vector<int>>& backwardTensors,
                                       const std::vector<int>& gradTensors) {
        OpConverter::Result result;
        result.newTensorOffset = net->tensorName.size();
        auto outputIndex       = forwardOp->outputIndexes[0];
        auto outputDiff        = backwardTensors.find(outputIndex)->second[0];

        unique_ptr<OpT> newOp(new OpT);
        newOp->name          = forwardOp->name + "_Grad";
        newOp->inputIndexes  = {forwardOp->inputIndexes[0], outputDiff};
        newOp->outputIndexes = {gradTensors[0]};
        newOp->type          = OpType_Relu6Grad;
        newOp->main.type     = OpParameter_NONE;
        result.opLists.emplace_back(std::move(newOp));

        return result;
    }
};
class ReluGradCreator : public OpGrad::Creator {
public:
    virtual OpGrad* onCreate(const MNN::OpT* op, const std::vector<MNN::Tensor*>& inputs,
                             const std::vector<MNN::Tensor*>& outputs) const override {
        return new ReluGrad;
    }
};
class Relu6GradCreator : public OpGrad::Creator {
public:
    virtual OpGrad* onCreate(const MNN::OpT* op, const std::vector<MNN::Tensor*>& inputs,
                             const std::vector<MNN::Tensor*>& outputs) const override {
        return new Relu6Grad;
    }
};

static const auto gRegister = []() {
    static ReluGradCreator _c;
    OpGrad::insert(OpType_ReLU, &_c);
    static Relu6GradCreator _d;
    OpGrad::insert(OpType_ReLU6, &_d);
    return true;
}();
