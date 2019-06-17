//
//  PoolGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "PoolGrad.hpp"
#include "Macro.h"
using namespace std;
using namespace MNN;

class PoolGrad : public OpGrad {
public:
    PoolGrad() {
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
        newOp->inputIndexes  = {forwardOp->inputIndexes[0], forwardOp->outputIndexes[0], outputDiff};
        newOp->outputIndexes = {gradTensors[0]};
        newOp->type          = OpType_PoolGrad;
        auto copyP           = new PoolT(*forwardOp->main.AsPool());
        newOp->main.type     = OpParameter_Pool;
        newOp->main.value    = copyP;
        result.opLists.emplace_back(std::move(newOp));
        return result;
    }
};
class PoolGradCreator : public OpGrad::Creator {
public:
    virtual OpGrad* onCreate(const MNN::OpT* op, const std::vector<MNN::Tensor*>& inputs,
                             const std::vector<MNN::Tensor*>& outputs) const override {
        return new PoolGrad;
    }
};
static const auto gRegister = []() {
    static PoolGradCreator _c;
    OpGrad::insert(OpType_Pooling, &_c);
    return true;
}();
