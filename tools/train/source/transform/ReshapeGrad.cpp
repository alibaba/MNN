//
//  ReshapeGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ReshapeGrad.hpp"
#include "Macro.h"
using namespace std;
using namespace MNN;

class ReshapeGrad : public OpGrad {
public:
    virtual OpConverter::Result onGrad(const MNN::NetT* net, const MNN::OpT* forwardOp,
                                       std::map<int, std::vector<int>>& backwardTensors,
                                       const std::vector<int>& gradTensors) {
        OpConverter::Result result;
        result.newTensorOffset = net->tensorName.size();
        // Create Shape Op and Tensor
        auto newTensorId = (int)net->tensorName.size();
        {
            unique_ptr<OpT> newOp(new OpT);
            newOp->name          = forwardOp->name + "__Input_Shape";
            newOp->inputIndexes  = {forwardOp->inputIndexes[0]};
            newOp->outputIndexes = {newTensorId};
            newOp->type          = OpType_Shape;
            result.tensorNames.emplace_back(newOp->name);
            result.opLists.emplace_back(std::move(newOp));
        }

        // Create Reshape Op
        {
            auto outputIndex = forwardOp->outputIndexes[0];
            auto outputDiff  = backwardTensors.find(outputIndex)->second[0];
            unique_ptr<OpT> newOp(new OpT);
            newOp->name                      = forwardOp->name + "__Grad";
            newOp->inputIndexes              = {outputDiff, newTensorId};
            newOp->outputIndexes             = {gradTensors[0]};
            newOp->type                      = OpType_Reshape;
            newOp->main.type                 = OpParameter_Reshape;
            newOp->main.value                = new ReshapeT;
            newOp->main.AsReshape()->dimType = MNN_DATA_FORMAT_NCHW;
            result.opLists.emplace_back(std::move(newOp));
        }

        return result;
    }
};

class ReshapeGradCreator : public OpGrad::Creator {
public:
    virtual OpGrad* onCreate(const MNN::OpT* op, const std::vector<MNN::Tensor*>& inputs,
                             const std::vector<MNN::Tensor*>& outputs) const override {
        return new ReshapeGrad;
    }
};
static const auto gRegister = []() {
    static ReshapeGradCreator _c;
    OpGrad::insert(OpType_Reshape, &_c);
    OpGrad::insert(OpType_Squeeze, &_c);
    return true;
}();
