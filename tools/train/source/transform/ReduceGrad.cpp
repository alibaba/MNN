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

class ReduceGrad : public OpGrad {
public:
    ReduceGrad(const std::vector<int>& dims) {
        mDims = dims;
    }
    virtual ~ReduceGrad() {
    }

protected:
    std::vector<int> mDims;
};
class ReduceSumGrad : public ReduceGrad {
public:
    ReduceSumGrad(const std::vector<int>& dims) : ReduceGrad(dims) {
    }
    virtual OpConverter::Result onGrad(const MNN::NetT* net, const MNN::OpT* forwardOp,
                                       std::map<int, std::vector<int>>& backwardTensors,
                                       const std::vector<int>& gradTensors) {
        OpConverter::Result result;
        result.newTensorOffset = net->tensorName.size();
        // Create Zero Op and Tensor
        auto newTensorId = (int)net->tensorName.size();
        auto zeroId      = newTensorId;
        {
            unique_ptr<OpT> newOp(new OpT);
            newOp->name          = forwardOp->name + "__Zero";
            newOp->inputIndexes  = {forwardOp->inputIndexes[0]};
            newOp->outputIndexes = {newTensorId};
            newOp->type          = OpType_ZerosLike;
            result.tensorNames.emplace_back(newOp->name);
            result.opLists.emplace_back(std::move(newOp));
        }
        auto outputIndex   = forwardOp->outputIndexes[0];
        auto outputDiff    = backwardTensors.find(outputIndex)->second[0];
        auto currentOutput = outputDiff;
        auto dim           = mDims;
        if (dim.size() > 0) {
            // Create Unsqueeze Op
            unique_ptr<OpT> newOp(new OpT);
            newOp->name                               = forwardOp->name + "__Unsqueeze";
            newOp->inputIndexes                       = {currentOutput};
            newOp->outputIndexes                      = {(int)net->tensorName.size() + 1};
            newOp->type                               = OpType_Unsqueeze;
            newOp->main.type                          = OpParameter_SqueezeParam;
            newOp->main.value                         = new SqueezeParamT;
            newOp->main.AsSqueezeParam()->squeezeDims = dim;
            currentOutput                             = newOp->outputIndexes[0];
            result.tensorNames.emplace_back(newOp->name);
            result.opLists.emplace_back(std::move(newOp));
        }

        // Create Binary Op
        {
            unique_ptr<OpT> newOp(new OpT);
            newOp->name                      = forwardOp->name + "__Grad";
            newOp->inputIndexes              = {zeroId, currentOutput};
            newOp->outputIndexes             = {gradTensors[0]};
            newOp->type                      = OpType_BinaryOp;
            newOp->main.type                 = OpParameter_BinaryOp;
            newOp->main.value                = new BinaryOpT;
            newOp->main.AsBinaryOp()->T      = DataType_DT_FLOAT;
            newOp->main.AsBinaryOp()->opType = BinaryOpOperation_ADD;
            result.opLists.emplace_back(std::move(newOp));
        }
        return result;
    }
};
class ReduceMeanGrad : public ReduceGrad {
public:
    ReduceMeanGrad(const std::vector<int>& dims, const std::vector<Tensor*>& inputs) : ReduceGrad(dims) {
        auto input = inputs[0];
        float size = 1.0f;
        for (int i = 0; i < dims.size(); ++i) {
            size *= (float)input->length(i);
        }
        mScale = 1.0f / size;
    }
    virtual OpConverter::Result onGrad(const MNN::NetT* net, const MNN::OpT* forwardOp,
                                       std::map<int, std::vector<int>>& backwardTensors,
                                       const std::vector<int>& gradTensors) {
        OpConverter::Result result;
        result.newTensorOffset = net->tensorName.size();
        // Create Shape Op
        auto shapeId = result.newTensorOffset + 0;
        {
            unique_ptr<OpT> newOp(new OpT);
            newOp->name          = forwardOp->name + "__Shape";
            newOp->inputIndexes  = {forwardOp->inputIndexes[0]};
            newOp->outputIndexes = {shapeId};
            newOp->type          = OpType_Shape;
            result.tensorNames.emplace_back(newOp->name);
            result.opLists.emplace_back(std::move(newOp));
        }
        auto scaleId = result.newTensorOffset + 1;
        // Create scale
        {
            unique_ptr<OpT> newOp(new OpT);
            newOp->name                      = forwardOp->name + "__ScaleConst";
            newOp->inputIndexes              = {};
            newOp->outputIndexes             = {scaleId};
            newOp->type                      = OpType_Const;
            newOp->main.type                 = OpParameter_Blob;
            newOp->main.value                = new BlobT;
            newOp->main.AsBlob()->dataType   = DataType_DT_FLOAT;
            newOp->main.AsBlob()->float32s   = {mScale};
            newOp->main.AsBlob()->dataFormat = MNN_DATA_FORMAT_NHWC;
            result.tensorNames.emplace_back(newOp->name);
            result.opLists.emplace_back(std::move(newOp));
        }
        // Create Fill
        auto fillId = result.newTensorOffset + 2;
        {
            unique_ptr<OpT> newOp(new OpT);
            newOp->name          = forwardOp->name + "__Fill";
            newOp->inputIndexes  = {shapeId, scaleId};
            newOp->outputIndexes = {fillId};
            newOp->type          = OpType_Fill;
            result.tensorNames.emplace_back(newOp->name);
            result.opLists.emplace_back(std::move(newOp));
        }
        auto zeroId = fillId;

        auto outputIndex   = forwardOp->outputIndexes[0];
        auto outputDiff    = backwardTensors.find(outputIndex)->second[0];
        auto currentOutput = outputDiff;
        auto dim           = mDims;
        if (dim.size() > 0) {
            // Create Unsqueeze Op
            unique_ptr<OpT> newOp(new OpT);
            newOp->name                               = forwardOp->name + "__Unsqueeze";
            newOp->inputIndexes                       = {currentOutput};
            newOp->outputIndexes                      = {result.newTensorOffset + 3};
            newOp->type                               = OpType_Unsqueeze;
            newOp->main.type                          = OpParameter_SqueezeParam;
            newOp->main.value                         = new SqueezeParamT;
            newOp->main.AsSqueezeParam()->squeezeDims = dim;
            currentOutput                             = newOp->outputIndexes[0];
            result.tensorNames.emplace_back(newOp->name);
            result.opLists.emplace_back(std::move(newOp));
        }

        // Create Binary Op
        {
            unique_ptr<OpT> newOp(new OpT);
            newOp->name                      = forwardOp->name + "__Grad";
            newOp->inputIndexes              = {zeroId, currentOutput};
            newOp->outputIndexes             = {gradTensors[0]};
            newOp->type                      = OpType_BinaryOp;
            newOp->main.type                 = OpParameter_BinaryOp;
            newOp->main.value                = new BinaryOpT;
            newOp->main.AsBinaryOp()->T      = DataType_DT_FLOAT;
            newOp->main.AsBinaryOp()->opType = BinaryOpOperation_ADD;
            result.opLists.emplace_back(std::move(newOp));
        }
        return result;
    }

private:
    float mScale;
};
class ReduceGradCreator : public OpGrad::Creator {
public:
    virtual OpGrad* onCreate(const MNN::OpT* op, const std::vector<MNN::Tensor*>& inputs,
                             const std::vector<MNN::Tensor*>& outputs) const override {
        auto reduct = op->main.AsReductionParam();
        auto dims   = reduct->dim;
        if (inputs.size() >= 2) {
            auto inputDim = inputs[1];
            dims.resize(inputDim->elementSize());
            for (int i = 0; i < dims.size(); ++i) {
                dims[i] = inputDim->host<int32_t>()[i];
            }
        }
        // FUNC_PRINT(reduct->operation);
        if (reduct->operation == ReductionType_SUM) {
            return new ReduceSumGrad(dims);
        }
        if (reduct->operation == ReductionType_MEAN) {
            return new ReduceMeanGrad(dims, inputs);
        }
        return nullptr;
    }
};
static const auto gRegister = []() {
    static ReduceGradCreator _c;
    OpGrad::insert(OpType_Reduction, &_c);
    return true;
}();
