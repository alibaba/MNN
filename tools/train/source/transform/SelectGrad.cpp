//
//  SelectGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SelectGrad.hpp"
#include "Macro.h"
using namespace std;
using namespace MNN;

class SelectGrad : public OpGrad {
public:
    SelectGrad() {
        mType = SEMI_LINEAR;
    }
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
        // d (select(x, a, b)) = da * (x>0) + db * (x < 0)
        {
            // Cast x>0 -> float
            unique_ptr<OpT> mask(new OpT);
            mask->type                     = OpType_Cast;
            mask->main.type                = OpParameter_CastParam;
            mask->main.value               = new CastParamT;
            mask->main.AsCastParam()->dstT = DataType_DT_FLOAT;
            mask->main.AsCastParam()->srcT = DataType_DT_BOOL;
            mask->name                     = "Mask_" + forwardOp->name;
            mask->outputIndexes            = {result.newTensorOffset + 0};
            mask->inputIndexes             = {forwardOp->inputIndexes[0]};
            result.tensorNames.emplace_back(mask->name);
            result.opLists.emplace_back(std::move(mask));

            // da * (x>0)
            unique_ptr<OpT> mulDA(new OpT);
            mulDA->type                      = OpType_BinaryOp;
            mulDA->name                      = forwardOp->name + "_Grad_1";
            mulDA->outputIndexes             = {gradTensors[0]};
            mulDA->inputIndexes              = {outputDiff, result.newTensorOffset + 0};
            mulDA->main.type                 = OpParameter_BinaryOp;
            mulDA->main.value                = new BinaryOpT;
            mulDA->main.AsBinaryOp()->T      = DataType_DT_FLOAT;
            mulDA->main.AsBinaryOp()->opType = BinaryOpOperation_MUL;
            result.opLists.emplace_back(std::move(mulDA));

            // db * -((x>0)-1)
            {
                unique_ptr<OpT> one(new OpT);
                one->type                      = OpType_Const;
                one->name                      = forwardOp->name + "_One";
                one->outputIndexes             = {result.newTensorOffset + 1};
                one->inputIndexes              = {};
                one->main.type                 = OpParameter_Blob;
                one->main.value                = new BlobT;
                one->main.AsBlob()->dataType   = DataType_DT_FLOAT;
                one->main.AsBlob()->dataFormat = MNN_DATA_FORMAT_NHWC;
                one->main.AsBlob()->float32s   = {1.0f};
                result.tensorNames.emplace_back(one->name);
                result.opLists.emplace_back(std::move(one));

                unique_ptr<OpT> sub(new OpT);
                sub->type                      = OpType_BinaryOp;
                sub->name                      = forwardOp->name + "_SubMask";
                sub->outputIndexes             = {result.newTensorOffset + 2};
                sub->inputIndexes              = {result.newTensorOffset + 0, result.newTensorOffset + 1};
                sub->main.type                 = OpParameter_BinaryOp;
                sub->main.value                = new BinaryOpT;
                sub->main.AsBinaryOp()->T      = DataType_DT_FLOAT;
                sub->main.AsBinaryOp()->opType = BinaryOpOperation_SUB;
                result.tensorNames.emplace_back(sub->name);
                result.opLists.emplace_back(std::move(sub));

                unique_ptr<OpT> neg(new OpT);
                neg->type                     = OpType_UnaryOp;
                neg->name                     = forwardOp->name + "_NegMask";
                neg->outputIndexes            = {result.newTensorOffset + 3};
                neg->inputIndexes             = {result.newTensorOffset + 2};
                neg->main.type                = OpParameter_UnaryOp;
                neg->main.value               = new UnaryOpT;
                neg->main.AsUnaryOp()->T      = DataType_DT_FLOAT;
                neg->main.AsUnaryOp()->opType = UnaryOpOperation_NEG;
                result.tensorNames.emplace_back(neg->name);
                result.opLists.emplace_back(std::move(neg));

                unique_ptr<OpT> mulDB(new OpT);
                mulDB->type                      = OpType_BinaryOp;
                mulDB->name                      = forwardOp->name + "_Grad_2";
                mulDB->outputIndexes             = {gradTensors[1]};
                mulDB->inputIndexes              = {outputDiff, result.newTensorOffset + 3};
                mulDB->main.type                 = OpParameter_BinaryOp;
                mulDB->main.value                = new BinaryOpT;
                mulDB->main.AsBinaryOp()->T      = DataType_DT_FLOAT;
                mulDB->main.AsBinaryOp()->opType = BinaryOpOperation_MUL;
                result.opLists.emplace_back(std::move(mulDB));
            }
        }

        return result;
    }

    virtual bool onGradCommon(MNN::NetT* dest, const MNN::OpT* forwardOp,
                              std::map<int, std::vector<int>>& backwardTensors) override {
        // Create New Diff Tensors
        std::vector<int> gradTensors;
        for (int i = 1; i < forwardOp->inputIndexes.size(); ++i) {
            int newTensorId = dest->tensorName.size();
            gradTensors.emplace_back(newTensorId);
            dest->tensorName.emplace_back(forwardOp->name + "_" + numberToString(i));
            auto inputIndex = forwardOp->inputIndexes[i];
            if (backwardTensors.find(inputIndex) == backwardTensors.end()) {
                backwardTensors.insert(make_pair(inputIndex, vector<int>{}));
            }
            backwardTensors[inputIndex].emplace_back(newTensorId);
        }
        auto result = this->onGrad(dest, forwardOp, backwardTensors, gradTensors);
        dest->tensorName.insert(dest->tensorName.end(), result.tensorNames.begin(), result.tensorNames.end());
        for (auto& op : result.opLists) {
            dest->oplists.emplace_back(std::move(op));
        }
        return true;
    }
};

class SelectGradCreator : public OpGrad::Creator {
public:
    virtual OpGrad* onCreate(const MNN::OpT* op, const std::vector<MNN::Tensor*>& inputs,
                             const std::vector<MNN::Tensor*>& outputs) const override {
        return new SelectGrad;
    }
};
static const auto gRegister = []() {
    static SelectGradCreator _c;
    OpGrad::insert(OpType_Select, &_c);
    return true;
}();
