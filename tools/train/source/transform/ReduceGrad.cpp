//
//  ReduceGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "OpGrad.hpp"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class ReduceGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr, const std::vector<Express::VARP>& output, const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result;
        auto inputs = expr->inputs();
        result.resize(inputs.size());
        std::unique_ptr<OpT> forwardOp(expr->get()->UnPack());
        std::vector<int> dim = forwardOp->main.AsReductionParam()->dim;
        auto keepDim = forwardOp->main.AsReductionParam()->keepDims;
        if (inputs.size() > 1) {
            dim.clear();
            auto ptr = inputs[1]->readMap<int32_t>();
            auto shape = inputs[1]->getInfo();
            for (int i=0; i<shape->size; ++i) {
                dim.emplace_back(ptr[i]);
            }
            inputs[1]->unMap();
        }
        if (dim.empty()) {
            auto shape = inputs[0]->getInfo();
            for (int i=0; i<shape->dim.size(); ++i) {
                dim.emplace_back(i);
            }
        }
        if (forwardOp->main.AsReductionParam()->operation == ReductionType_SUM) {
            VARP init;
            {
                unique_ptr<OpT> newOp(new OpT);
                newOp->name          = forwardOp->name + "__Zero";
                newOp->type          = OpType_ZerosLike;
                init = Variable::create(Expr::create(std::move(newOp), {inputs[0]}));
            }
            auto outputDiff    = backwardOutput[0];
            auto currentOutput = outputDiff;
            if (!keepDim) {
                // Create Unsqueeze Op
                unique_ptr<OpT> newOp(new OpT);
                newOp->name                               = forwardOp->name + "__Unsqueeze";
                newOp->type                               = OpType_Unsqueeze;
                newOp->main.type                          = OpParameter_SqueezeParam;
                newOp->main.value                         = new SqueezeParamT;
                newOp->main.AsSqueezeParam()->squeezeDims = dim;
                outputDiff = Variable::create(Expr::create(std::move(newOp), {outputDiff}));
            }
            result[0] = _Add(init, outputDiff);
        }
        return result;
    }
};
//class ReduceMeanGrad : public ReduceGrad {
//public:
//    ReduceMeanGrad(const std::vector<int>& dims, const std::vector<Tensor*>& inputs) : ReduceGrad(dims) {
//        auto input = inputs[0];
//        float size = 1.0f;
//        for (int i = 0; i < dims.size(); ++i) {
//            size *= (float)input->length(i);
//        }
//        mScale = 1.0f / size;
//    }
//    virtual OpConverter::Result onGrad(const MNN::NetT* net, const MNN::OpT* forwardOp,
//                                       std::map<int, std::vector<int>>& backwardTensors,
//                                       const std::vector<int>& gradTensors) {
//        OpConverter::Result result;
//        result.newTensorOffset = net->tensorName.size();
//        // Create Shape Op
//        auto shapeId = result.newTensorOffset + 0;
//        {
//            unique_ptr<OpT> newOp(new OpT);
//            newOp->name          = forwardOp->name + "__Shape";
//            newOp->inputIndexes  = {forwardOp->inputIndexes[0]};
//            newOp->outputIndexes = {shapeId};
//            newOp->type          = OpType_Shape;
//            result.tensorNames.emplace_back(newOp->name);
//            result.opLists.emplace_back(std::move(newOp));
//        }
//        auto scaleId = result.newTensorOffset + 1;
//        // Create scale
//        {
//            unique_ptr<OpT> newOp(new OpT);
//            newOp->name                      = forwardOp->name + "__ScaleConst";
//            newOp->inputIndexes              = {};
//            newOp->outputIndexes             = {scaleId};
//            newOp->type                      = OpType_Const;
//            newOp->main.type                 = OpParameter_Blob;
//            newOp->main.value                = new BlobT;
//            newOp->main.AsBlob()->dataType   = DataType_DT_FLOAT;
//            newOp->main.AsBlob()->float32s   = {mScale};
//            newOp->main.AsBlob()->dataFormat = MNN_DATA_FORMAT_NHWC;
//            result.tensorNames.emplace_back(newOp->name);
//            result.opLists.emplace_back(std::move(newOp));
//        }
//        // Create Fill
//        auto fillId = result.newTensorOffset + 2;
//        {
//            unique_ptr<OpT> newOp(new OpT);
//            newOp->name          = forwardOp->name + "__Fill";
//            newOp->inputIndexes  = {shapeId, scaleId};
//            newOp->outputIndexes = {fillId};
//            newOp->type          = OpType_Fill;
//            result.tensorNames.emplace_back(newOp->name);
//            result.opLists.emplace_back(std::move(newOp));
//        }
//        auto zeroId = fillId;
//
//        auto outputIndex   = forwardOp->outputIndexes[0];
//        auto outputDiff    = backwardTensors.find(outputIndex)->second[0];
//        auto currentOutput = outputDiff;
//        auto dim           = mDims;
//        if (dim.size() > 0) {
//            // Create Unsqueeze Op
//            unique_ptr<OpT> newOp(new OpT);
//            newOp->name                               = forwardOp->name + "__Unsqueeze";
//            newOp->inputIndexes                       = {currentOutput};
//            newOp->outputIndexes                      = {result.newTensorOffset + 3};
//            newOp->type                               = OpType_Unsqueeze;
//            newOp->main.type                          = OpParameter_SqueezeParam;
//            newOp->main.value                         = new SqueezeParamT;
//            newOp->main.AsSqueezeParam()->squeezeDims = dim;
//            currentOutput                             = newOp->outputIndexes[0];
//            result.tensorNames.emplace_back(newOp->name);
//            result.opLists.emplace_back(std::move(newOp));
//        }
//
//        // Create Binary Op
//        {
//            unique_ptr<OpT> newOp(new OpT);
//            newOp->name                      = forwardOp->name + "__Grad";
//            newOp->inputIndexes              = {zeroId, currentOutput};
//            newOp->outputIndexes             = {gradTensors[0]};
//            newOp->type                      = OpType_BinaryOp;
//            newOp->main.type                 = OpParameter_BinaryOp;
//            newOp->main.value                = new BinaryOpT;
//            newOp->main.AsBinaryOp()->T      = DataType_DT_FLOAT;
//            newOp->main.AsBinaryOp()->opType = BinaryOpOperation_ADD;
//            result.opLists.emplace_back(std::move(newOp));
//        }
//        return result;
//    }
//
//private:
//    float mScale;
//};

static const auto gRegister = []() {
    static ReduceGrad _c;
    OpGrad::insert(OpType_Reduction, &_c);
    return true;
}();
