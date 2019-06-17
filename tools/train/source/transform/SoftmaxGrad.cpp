//
//  SoftmaxGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SoftmaxGrad.hpp"
#include "Macro.h"
using namespace std;
using namespace MNN;

class SoftmaxGrad : public OpGrad {
public:
    SoftmaxGrad() {
        mType = NO_LINEAR;
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
        newOp->inputIndexes  = {forwardOp->outputIndexes[0], outputDiff};
        newOp->outputIndexes = {gradTensors[0]};
        newOp->type          = OpType_SoftmaxGrad;
        newOp->main.type     = OpParameter_Axis;
        newOp->main.value    = new AxisT(*forwardOp->main.AsAxis());
        result.opLists.emplace_back(std::move(newOp));
        return result;
    }
};
class SoftmaxGradCreator : public OpGrad::Creator {
public:
    virtual OpGrad* onCreate(const MNN::OpT* op, const std::vector<MNN::Tensor*>& inputs,
                             const std::vector<MNN::Tensor*>& outputs) const override {
        return new SoftmaxGrad;
    }
};
static const auto gRegister = []() {
    static SoftmaxGradCreator _c;
    OpGrad::insert(OpType_Softmax, &_c);
    return true;
}();
