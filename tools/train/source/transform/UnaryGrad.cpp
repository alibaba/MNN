//
//  UnaryGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "OpGrad.hpp"
using namespace std;
using namespace MNN;

class UnaryGrad : public OpGrad {
public:
    virtual OpConverter::Result onGrad(const MNN::NetT* net, const MNN::OpT* forwardOp,
                                       std::map<int, std::vector<int>>& backwardTensors,
                                       const std::vector<int>& gradTensors) override {
        OpConverter::Result result;
        result.newTensorOffset = net->tensorName.size();
        auto outputIndex       = forwardOp->outputIndexes[0];
        auto outputDiffIter    = backwardTensors.find(outputIndex);
        if (outputDiffIter == backwardTensors.end()) {
            return result;
        }
        auto outputDiff = outputDiffIter->second[0];

        switch (forwardOp->main.AsUnaryOp()->opType) {
            case MNN::UnaryOpOperation_LOG1P: {
                // d log(1+x) = 1/(1+x) * dx = dx / (1+x)
                std::unique_ptr<OpT> oneConst(new OpT);
                oneConst->main.type                 = OpParameter_Blob;
                oneConst->main.value                = new BlobT;
                oneConst->type                      = OpType_Const;
                oneConst->main.AsBlob()->float32s   = {1.0f};
                oneConst->main.AsBlob()->dataFormat = MNN_DATA_FORMAT_NHWC;
                oneConst->main.AsBlob()->dataType   = DataType_DT_FLOAT;
                oneConst->outputIndexes             = {result.newTensorOffset + 0};
                oneConst->name                      = forwardOp->name + "_ConstOne";
                auto oneTensor                      = oneConst->outputIndexes[0];
                result.tensorNames.emplace_back(oneConst->name);
                result.opLists.emplace_back(std::move(oneConst));

                std::unique_ptr<OpT> addOne(new OpT);
                addOne->main.type     = OpParameter_BinaryOp;
                addOne->main.value    = new BinaryOpT;
                addOne->type          = OpType_BinaryOp;
                addOne->inputIndexes  = {forwardOp->inputIndexes[0], oneTensor};
                addOne->outputIndexes = {result.newTensorOffset + 1};
                addOne->name          = forwardOp->name + "_AddOne";
                result.tensorNames.emplace_back(addOne->name);
                addOne->main.AsBinaryOp()->opType = BinaryOpOperation_ADD;
                result.opLists.emplace_back(std::move(addOne));

                std::unique_ptr<OpT> div(new OpT);
                div->main.type                 = OpParameter_BinaryOp;
                div->main.value                = new BinaryOpT;
                div->type                      = OpType_BinaryOp;
                div->inputIndexes              = {outputDiff, result.newTensorOffset + 1};
                div->outputIndexes             = {gradTensors[0]};
                div->name                      = forwardOp->name + "_Div";
                div->main.AsBinaryOp()->opType = BinaryOpOperation_REALDIV;
                result.opLists.emplace_back(std::move(div));
            } break;
            case MNN::UnaryOpOperation_EXP: {
                // d Exp(x) = Exp(x) * dx
                std::unique_ptr<OpT> mul(new OpT);
                mul->main.type                 = OpParameter_BinaryOp;
                mul->main.value                = new BinaryOpT;
                mul->type                      = OpType_BinaryOp;
                mul->inputIndexes              = {outputDiff, forwardOp->outputIndexes[0]};
                mul->outputIndexes             = {gradTensors[0]};
                mul->main.AsBinaryOp()->opType = BinaryOpOperation_MUL;
                mul->name                      = forwardOp->name + "_Grad";
                result.opLists.emplace_back(std::move(mul));
            } break;
            case MNN::UnaryOpOperation_LOG: {
                // d Log(x) =  dx / x
                std::unique_ptr<OpT> mul(new OpT);
                mul->main.type                 = OpParameter_BinaryOp;
                mul->main.value                = new BinaryOpT;
                mul->type                      = OpType_BinaryOp;
                mul->inputIndexes              = {outputDiff, forwardOp->inputIndexes[0]};
                mul->outputIndexes             = {gradTensors[0]};
                mul->main.AsBinaryOp()->opType = BinaryOpOperation_REALDIV;
                mul->name                      = forwardOp->name + "_Grad";
                result.opLists.emplace_back(std::move(mul));
            } break;
            case MNN::UnaryOpOperation_NEG: {
                // d (-x) = - dx
                std::unique_ptr<OpT> Neg(new OpT);
                Neg->main.type                = OpParameter_UnaryOp;
                Neg->main.value               = new UnaryOpT;
                Neg->type                     = OpType_UnaryOp;
                Neg->inputIndexes             = {outputDiff};
                Neg->outputIndexes            = {gradTensors[0]};
                Neg->main.AsUnaryOp()->opType = UnaryOpOperation_NEG;
                Neg->name                     = forwardOp->name + "_Grad";
                result.opLists.emplace_back(std::move(Neg));
            } break;
            case MNN::UnaryOpOperation_SQRT: {
                // d (-sqrt(x)) = 0.5 / sqrt(x) * dx
                std::unique_ptr<OpT> oneConst(new OpT);
                oneConst->main.type                 = OpParameter_Blob;
                oneConst->main.value                = new BlobT;
                oneConst->type                      = OpType_Const;
                oneConst->main.AsBlob()->float32s   = {0.5f};
                oneConst->main.AsBlob()->dataFormat = MNN_DATA_FORMAT_NHWC;
                oneConst->main.AsBlob()->dataType   = DataType_DT_FLOAT;
                oneConst->outputIndexes             = {result.newTensorOffset + 0};
                oneConst->name                      = forwardOp->name + "_ConstHalf";
                auto oneTensor                      = oneConst->outputIndexes[0];
                result.tensorNames.emplace_back(oneConst->name);
                result.opLists.emplace_back(std::move(oneConst));

                std::unique_ptr<OpT> mul(new OpT);
                mul->main.type                 = OpParameter_BinaryOp;
                mul->main.value                = new BinaryOpT;
                mul->type                      = OpType_BinaryOp;
                mul->inputIndexes              = {outputDiff, result.newTensorOffset + 0};
                mul->outputIndexes             = {result.newTensorOffset + 1};
                mul->main.AsBinaryOp()->opType = BinaryOpOperation_MUL;
                mul->name                      = forwardOp->name + "_Mul";
                result.tensorNames.emplace_back(mul->name);
                result.opLists.emplace_back(std::move(mul));

                std::unique_ptr<OpT> div(new OpT);
                div->main.type                 = OpParameter_BinaryOp;
                div->main.value                = new BinaryOpT;
                div->type                      = OpType_BinaryOp;
                div->inputIndexes              = {result.newTensorOffset + 1, forwardOp->outputIndexes[0]};
                div->outputIndexes             = {gradTensors[0]};
                div->main.AsBinaryOp()->opType = BinaryOpOperation_REALDIV;
                div->name                      = forwardOp->name + "_Grad";
                result.opLists.emplace_back(std::move(div));

            } break;
            case MNN::UnaryOpOperation_SQUARE: {
                // d (x^2) = (x*dx + x*dx)
                std::unique_ptr<OpT> mul(new OpT);
                mul->main.type                 = OpParameter_BinaryOp;
                mul->main.value                = new BinaryOpT;
                mul->type                      = OpType_BinaryOp;
                mul->inputIndexes              = {outputDiff, forwardOp->inputIndexes[0]};
                mul->outputIndexes             = {result.newTensorOffset};
                mul->main.AsBinaryOp()->opType = BinaryOpOperation_MUL;
                mul->main.AsBinaryOp()->T      = DataType_DT_FLOAT;
                mul->name                      = forwardOp->name + "_Grad_mul";
                result.tensorNames.emplace_back(mul->name);
                result.opLists.emplace_back(std::move(mul));

                std::unique_ptr<OpT> add(new OpT);
                add->main.type                 = OpParameter_BinaryOp;
                add->main.value                = new BinaryOpT;
                add->type                      = OpType_BinaryOp;
                add->inputIndexes              = {result.newTensorOffset, result.newTensorOffset};
                add->outputIndexes             = {gradTensors[0]};
                add->main.AsBinaryOp()->opType = BinaryOpOperation_ADD;
                add->main.AsBinaryOp()->T      = DataType_DT_FLOAT;
                add->name                      = forwardOp->name + "_Grad";
                result.opLists.emplace_back(std::move(add));
            } break;
            default:
                MNN_ASSERT(false);
                break;
        }

        return result;
    }
    virtual bool onGradCommon(MNN::NetT* net, const MNN::OpT* op,
                              std::map<int, std::vector<int>>& backwardTensors) override {
        return OpGrad::onGradCommon(net, op, backwardTensors);
    }
};
class SigmoidGrad : public OpGrad {
public:
    virtual OpConverter::Result onGrad(const MNN::NetT* net, const MNN::OpT* forwardOp,
                                       std::map<int, std::vector<int>>& backwardTensors,
                                       const std::vector<int>& gradTensors) override {
        OpConverter::Result result;
        result.newTensorOffset = net->tensorName.size();
        auto outputIndex       = forwardOp->outputIndexes[0];
        auto outputDiffIter    = backwardTensors.find(outputIndex);
        if (outputDiffIter == backwardTensors.end()) {
            return result;
        }
        auto outputDiff = outputDiffIter->second[0];

        // y = (1/(1+e(-x))) , dy = y(1-y) * dx = (y*y - y)*dx
        std::unique_ptr<OpT> mul(new OpT);
        mul->main.type                 = OpParameter_BinaryOp;
        mul->main.value                = new BinaryOpT;
        mul->type                      = OpType_BinaryOp;
        mul->inputIndexes              = {forwardOp->outputIndexes[0], forwardOp->outputIndexes[0]};
        mul->outputIndexes             = {result.newTensorOffset + 0};
        mul->main.AsBinaryOp()->opType = BinaryOpOperation_MUL;
        mul->name                      = forwardOp->name + "_Square";
        result.tensorNames.emplace_back(mul->name);
        result.opLists.emplace_back(std::move(mul));

        std::unique_ptr<OpT> sub(new OpT);
        sub->main.type                 = OpParameter_BinaryOp;
        sub->main.value                = new BinaryOpT;
        sub->type                      = OpType_BinaryOp;
        sub->inputIndexes              = {result.newTensorOffset + 0, forwardOp->outputIndexes[0]};
        sub->outputIndexes             = {result.newTensorOffset + 1};
        sub->main.AsBinaryOp()->opType = BinaryOpOperation_SUB;
        sub->name                      = forwardOp->name + "_Sub";
        result.tensorNames.emplace_back(sub->name);
        result.opLists.emplace_back(std::move(sub));

        std::unique_ptr<OpT> grad(new OpT);
        grad->main.type                 = OpParameter_BinaryOp;
        grad->main.value                = new BinaryOpT;
        grad->type                      = OpType_BinaryOp;
        grad->inputIndexes              = {result.newTensorOffset + 1, outputDiff};
        grad->outputIndexes             = {gradTensors[0]};
        grad->main.AsBinaryOp()->opType = BinaryOpOperation_MUL;
        grad->name                      = forwardOp->name + "_Grad";
        result.tensorNames.emplace_back(grad->name);
        result.opLists.emplace_back(std::move(grad));
        return result;
    }
};

class UnaryGradCreator : public OpGrad::Creator {
public:
    virtual OpGrad* onCreate(const MNN::OpT* op, const std::vector<MNN::Tensor*>& inputs,
                             const std::vector<MNN::Tensor*>& outputs) const override {
        if (op->type == OpType_Sigmoid) {
            return new SigmoidGrad;
        }
        return new UnaryGrad;
    }
};
static const auto gRegister = []() {
    static UnaryGradCreator _c;
    OpGrad::insert(OpType_UnaryOp, &_c);
    OpGrad::insert(OpType_Sigmoid, &_c);
    return true;
}();
