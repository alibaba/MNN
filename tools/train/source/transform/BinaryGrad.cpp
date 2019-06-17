//
//  BinaryGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "BinaryGrad.hpp"
#include "Macro.h"
using namespace std;
using namespace MNN;

class BinaryGrad : public OpGrad {
public:
    BinaryGrad(const std::vector<MNN::Tensor*>& inputs, const std::vector<MNN::Tensor*>& outputs) {
        mInputs  = inputs;
        mOutputs = outputs;
    }
    virtual OpConverter::Result onGrad(const MNN::NetT* net, const MNN::OpT* forwardOp,
                                       std::map<int, std::vector<int>>& backwardTensors,
                                       const std::vector<int>& gradTensors) override {
        return OpConverter::Result();
    }
    virtual bool onGradCommon(MNN::NetT* net, const MNN::OpT* op,
                              std::map<int, std::vector<int>>& backwardTensors) override {
        auto outputDiffIter = backwardTensors.find(op->outputIndexes[0]);
        if (outputDiffIter == backwardTensors.end()) {
            return false;
        }
        OpConverter::Result result;
        std::map<int, std::vector<int>> newBackwardTensors;
        std::map<int, Tensor*> inputTensors;
        for (int i = 0; i < op->inputIndexes.size(); ++i) {
            inputTensors[op->inputIndexes[i]]       = mInputs[i];
            newBackwardTensors[op->inputIndexes[i]] = std::vector<int>{};
        }
        result.newTensorOffset = (int)net->tensorName.size();
        auto outputDiff        = outputDiffIter->second[0];
        switch (op->main.AsBinaryOp()->opType) {
            case BinaryOpOperation_ADD: {
                for (auto inputIndex : op->inputIndexes) {
                    newBackwardTensors[inputIndex].emplace_back(outputDiff);
                }
                break;
            }
            case BinaryOpOperation_SUB: {
                MNN_ASSERT(2 == op->inputIndexes.size());
                newBackwardTensors[op->inputIndexes[0]].emplace_back(outputDiff);
                {
                    // Neg
                    auto newTensorId = result.newTensorOffset;
                    auto newOpName   = op->name + "_Neg";
                    std::unique_ptr<OpT> newOp(new OpT);
                    newOp->name                     = newOpName;
                    newOp->inputIndexes             = {outputDiff};
                    newOp->outputIndexes            = {(int)newTensorId};
                    newOp->type                     = OpType_UnaryOp;
                    newOp->main.type                = OpParameter_UnaryOp;
                    newOp->main.value               = new UnaryOpT;
                    newOp->main.AsUnaryOp()->opType = UnaryOpOperation_NEG;
                    result.tensorNames.emplace_back(newOpName);
                    result.opLists.emplace_back(std::move(newOp));

                    newBackwardTensors[op->inputIndexes[1]].emplace_back(newTensorId);
                }
                break;
            }
            case BinaryOpOperation_MUL: {
                MNN_ASSERT(2 == op->inputIndexes.size());
                for (int i = 0; i < 2; ++i) {
                    auto inputIndex  = op->inputIndexes[i];
                    auto newTensorId = result.newTensorOffset + i;
                    auto newOpName   = op->name + "_Grad_" + numberToString(i);
                    std::unique_ptr<OpT> newOp(new OpT);
                    newOp->name                      = newOpName;
                    newOp->inputIndexes              = {outputDiff, op->inputIndexes[1 - i]};
                    newOp->outputIndexes             = {(int)newTensorId};
                    newOp->type                      = OpType_BinaryOp;
                    newOp->main.type                 = OpParameter_BinaryOp;
                    newOp->main.value                = new BinaryOpT;
                    newOp->main.AsBinaryOp()->opType = BinaryOpOperation_MUL;

                    result.tensorNames.emplace_back(newOpName);
                    result.opLists.emplace_back(std::move(newOp));
                    newBackwardTensors[inputIndex].emplace_back(newTensorId);
                }
                break;
            }
            case BinaryOpOperation_REALDIV: {
                // d (u / v) = dx / v , -dx*u(1/v)*(1/v)
                MNN_ASSERT(2 == op->inputIndexes.size());
                {
                    auto inputIndex  = op->inputIndexes[0];
                    auto newTensorId = result.newTensorOffset + 0;
                    auto newOpName   = op->name + "_Grad_0";
                    std::unique_ptr<OpT> newOp(new OpT);
                    newOp->name                      = newOpName;
                    newOp->inputIndexes              = {outputDiff, op->inputIndexes[1]};
                    newOp->outputIndexes             = {(int)newTensorId};
                    newOp->type                      = OpType_BinaryOp;
                    newOp->main.type                 = OpParameter_BinaryOp;
                    newOp->main.value                = new BinaryOpT;
                    newOp->main.AsBinaryOp()->opType = BinaryOpOperation_REALDIV;

                    result.tensorNames.emplace_back(newOpName);
                    result.opLists.emplace_back(std::move(newOp));
                    newBackwardTensors[inputIndex].emplace_back(newTensorId);
                }
                {
                    {
                        // u(1/v)*(1/v)
                        auto newTensorId = result.newTensorOffset + 1;
                        auto newOpName   = op->name + "_DIV_1";
                        std::unique_ptr<OpT> newOp(new OpT);
                        newOp->name                      = newOpName;
                        newOp->inputIndexes              = {op->outputIndexes[0], op->inputIndexes[1]};
                        newOp->outputIndexes             = {(int)newTensorId};
                        newOp->type                      = OpType_BinaryOp;
                        newOp->main.type                 = OpParameter_BinaryOp;
                        newOp->main.value                = new BinaryOpT;
                        newOp->main.AsBinaryOp()->opType = BinaryOpOperation_REALDIV;

                        result.tensorNames.emplace_back(newOpName);
                        result.opLists.emplace_back(std::move(newOp));
                    }
                    {
                        // dx*u(1/v)*(1/v)
                        auto newTensorId = result.newTensorOffset + 2;
                        auto newOpName   = op->name + "_MUL_1";
                        std::unique_ptr<OpT> newOp(new OpT);
                        newOp->name                      = newOpName;
                        newOp->inputIndexes              = {result.newTensorOffset + 1, outputDiff};
                        newOp->outputIndexes             = {(int)newTensorId};
                        newOp->type                      = OpType_BinaryOp;
                        newOp->main.type                 = OpParameter_BinaryOp;
                        newOp->main.value                = new BinaryOpT;
                        newOp->main.AsBinaryOp()->opType = BinaryOpOperation_MUL;

                        result.tensorNames.emplace_back(newOpName);
                        result.opLists.emplace_back(std::move(newOp));
                    }
                    {
                        // dx*u(1/v)*(1/v)
                        auto newTensorId = result.newTensorOffset + 3;
                        auto newOpName   = op->name + "_NEG_1";
                        std::unique_ptr<OpT> newOp(new OpT);
                        newOp->name                     = newOpName;
                        newOp->inputIndexes             = {result.newTensorOffset + 2};
                        newOp->outputIndexes            = {(int)newTensorId};
                        newOp->type                     = OpType_UnaryOp;
                        newOp->main.type                = OpParameter_UnaryOp;
                        newOp->main.value               = new UnaryOpT;
                        newOp->main.AsUnaryOp()->opType = UnaryOpOperation_NEG;
                        newOp->main.AsUnaryOp()->T      = DataType_DT_FLOAT;

                        result.tensorNames.emplace_back(newOpName);
                        result.opLists.emplace_back(std::move(newOp));
                        newBackwardTensors[op->inputIndexes[1]].emplace_back(newTensorId);
                    }
                    break;
                }
            }
            default:
                break;
        }

        if (newBackwardTensors.empty()) {
            return false;
        }
        auto output = mOutputs[0];
        // Add reduce if necessary
        for (auto& iter : newBackwardTensors) {
            auto input = inputTensors[iter.first];
            if (input->dimensions() != output->dimensions()) {
                for (int v = 0; v < iter.second.size(); ++v) {
                    std::unique_ptr<OpT> reduct(new OpT);
                    reduct->type                               = OpType_Reduction;
                    int outputIndex                            = result.newTensorOffset + result.tensorNames.size();
                    reduct->outputIndexes                      = {outputIndex};
                    reduct->inputIndexes                       = {iter.second[v]};
                    reduct->main.type                          = OpParameter_ReductionParam;
                    reduct->main.value                         = new ReductionParamT;
                    reduct->main.AsReductionParam()->keepDims  = false;
                    reduct->main.AsReductionParam()->dType     = DataType_DT_FLOAT;
                    reduct->main.AsReductionParam()->operation = ReductionType_SUM;
                    auto diff                                  = output->dimensions() - input->dimensions();
                    for (int d = 0; d < diff; ++d) {
                        reduct->main.AsReductionParam()->dim.emplace_back(d);
                    }
                    reduct->name = op->name + "_Reduce_" + numberToString(iter.second[v]);
                    result.tensorNames.emplace_back(reduct->name);
                    result.opLists.emplace_back(std::move(reduct));
                    iter.second[v] = outputIndex;
                }
            } else {
                auto dimension = input->dimensions();
                std::vector<int> reduceDims;
                for (int d = 0; d < dimension; ++d) {
                    if (input->length(d) != output->length(d)) {
                        reduceDims.emplace_back(d);
                    }
                }
                if (!reduceDims.empty()) {
                    for (int v = 0; v < iter.second.size(); ++v) {
                        std::unique_ptr<OpT> reduct(new OpT);
                        reduct->type                               = OpType_Reduction;
                        int outputIndex                            = result.newTensorOffset + result.tensorNames.size();
                        reduct->outputIndexes                      = {outputIndex};
                        reduct->inputIndexes                       = {iter.second[v]};
                        reduct->main.type                          = OpParameter_ReductionParam;
                        reduct->main.value                         = new ReductionParamT;
                        reduct->main.AsReductionParam()->keepDims  = false;
                        reduct->main.AsReductionParam()->dType     = DataType_DT_FLOAT;
                        reduct->main.AsReductionParam()->operation = ReductionType_SUM;
                        reduct->main.AsReductionParam()->keepDims  = true;
                        reduct->main.AsReductionParam()->dim       = reduceDims;
                        reduct->name = op->name + "_Reduce_" + numberToString(iter.second[v]);
                        result.tensorNames.emplace_back(reduct->name);
                        result.opLists.emplace_back(std::move(reduct));
                        iter.second[v] = outputIndex;
                    }
                }
            }
        }

        // Merge to Net
        for (int i = 0; i < result.tensorNames.size(); ++i) {
            net->tensorName.emplace_back(result.tensorNames[i]);
        }
        for (int i = 0; i < result.opLists.size(); ++i) {
            net->oplists.emplace_back(std::move(result.opLists[i]));
        }
        for (auto iter : newBackwardTensors) {
            if (backwardTensors.find(iter.first) == backwardTensors.end()) {
                backwardTensors.insert(std::make_pair(iter.first, std::vector<int>{}));
            }
            backwardTensors[iter.first].insert(backwardTensors[iter.first].begin(), iter.second.begin(),
                                               iter.second.end());
        }
        return true;
    }

private:
    std::vector<MNN::Tensor*> mInputs;
    std::vector<MNN::Tensor*> mOutputs;
};

class BinaryGradCreator : public OpGrad::Creator {
public:
    virtual OpGrad* onCreate(const MNN::OpT* op, const std::vector<MNN::Tensor*>& inputs,
                             const std::vector<MNN::Tensor*>& outputs) const override {
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        if (nullptr == input0 || nullptr == outputs[0] || nullptr == input1) {
            return nullptr;
        }
        return new BinaryGrad(inputs, outputs);
    }
};
static const auto gRegister = []() {
    static BinaryGradCreator _c;
    OpGrad::insert(OpType_BinaryOp, &_c);
    return true;
}();
