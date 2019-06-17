//
//  TensorConvertGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TensorConvertGrad.hpp"
#include "Macro.h"
using namespace std;
using namespace MNN;

class TensorConvertGrad : public OpGrad {
public:
    virtual OpConverter::Result onGrad(const MNN::NetT* net, const MNN::OpT* forwardOp,
                                       std::map<int, std::vector<int>>& backwardTensors,
                                       const std::vector<int>& gradTensors) {
        OpConverter::Result result;
        result.newTensorOffset = net->tensorName.size();
        auto outputIndex       = forwardOp->outputIndexes[0];
        auto outputDiff        = backwardTensors.find(outputIndex)->second[0];

        unique_ptr<OpT> newOp(new OpT);
        newOp->name          = forwardOp->name + "_Grad";
        newOp->inputIndexes  = {outputDiff};
        newOp->outputIndexes = {gradTensors[0]};
        newOp->type          = OpType_ConvertTensor;
        newOp->main.type     = OpParameter_TensorConvertInfo;
        auto cInfo           = new TensorConvertInfoT;
        cInfo->dest          = forwardOp->main.AsTensorConvertInfo()->source;
        cInfo->source        = forwardOp->main.AsTensorConvertInfo()->dest;
        newOp->main.value    = cInfo;
        result.opLists.emplace_back(std::move(newOp));
        return result;
    }
};
class TensorConvertGradCreator : public OpGrad::Creator {
public:
    virtual OpGrad* onCreate(const MNN::OpT* op, const std::vector<MNN::Tensor*>& inputs,
                             const std::vector<MNN::Tensor*>& outputs) const override {
        return new TensorConvertGrad;
    }
};
static const auto gRegister = []() {
    static TensorConvertGradCreator _c;
    OpGrad::insert(OpType_ConvertTensor, &_c);
    return true;
}();
