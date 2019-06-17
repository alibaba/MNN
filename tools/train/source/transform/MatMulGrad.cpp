//
//  MatMulGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MatMulGrad.hpp"
using namespace std;
using namespace MNN;

class MatMulGrad : public OpGrad {
public:
    MatMulGrad() {
        mType = LINEAR;
    }
    virtual OpConverter::Result onGrad(const MNN::NetT* net, const MNN::OpT* forwardOp,
                                       std::map<int, std::vector<int>>& backwardTensors,
                                       const std::vector<int>& gradTensors) {
        OpConverter::Result result;
        result.newTensorOffset = net->tensorName.size();
        auto outputIndex       = forwardOp->outputIndexes[0];
        auto outputDiffIter    = backwardTensors.find(outputIndex);
        if (outputDiffIter == backwardTensors.end()) {
            return result;
        }
        auto outputDiff = outputDiffIter->second[0];
        {
            // A' = C' * BT
            unique_ptr<OpT> newOp(new OpT);
            newOp->name                        = forwardOp->name + "_Grad0";
            newOp->inputIndexes                = {outputDiff, forwardOp->inputIndexes[1]};
            newOp->outputIndexes               = {gradTensors[0]};
            newOp->type                        = OpType_MatMul;
            newOp->main.type                   = OpParameter_MatMul;
            newOp->main.value                  = new MatMulT;
            newOp->main.AsMatMul()->transposeB = true;

            result.opLists.emplace_back(std::move(newOp));
        }
        {
            // B' = AT * C'
            unique_ptr<OpT> newOp(new OpT);
            newOp->name                        = forwardOp->name + "_Grad1";
            newOp->inputIndexes                = {forwardOp->inputIndexes[0], outputDiff};
            newOp->outputIndexes               = {gradTensors[1]};
            newOp->type                        = OpType_MatMul;
            newOp->main.type                   = OpParameter_MatMul;
            newOp->main.value                  = new MatMulT;
            newOp->main.AsMatMul()->transposeA = true;

            result.opLists.emplace_back(std::move(newOp));
        }
        return result;
    }
};
class MatMulGradCreator : public OpGrad::Creator {
public:
    virtual OpGrad* onCreate(const MNN::OpT* op, const std::vector<MNN::Tensor*>& inputs,
                             const std::vector<MNN::Tensor*>& outputs) const override {
        return new MatMulGrad;
    }
};
static const auto gRegister = []() {
    static MatMulGradCreator _c;
    OpGrad::insert(OpType_MatMul, &_c);
    return true;
}();
