//
//  ConvGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvGrad.hpp"
#include "Macro.h"
using namespace std;
using namespace MNN;

static const OpT* _findOpFromOutput(const NetT* net, int outputIndex) {
    for (auto& op : net->oplists) {
        for (auto output : op->outputIndexes) {
            if (output == outputIndex) {
                return op.get();
            }
        }
    }
    return nullptr;
}
class ConvGrad : public OpGrad {
public:
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
            // Create Zero Bias
            auto originConstOp = _findOpFromOutput(net, forwardOp->inputIndexes[2]);
            unique_ptr<OpT> newConstBias(new OpT);
            auto zeroBiasId    = result.newTensorOffset + 0;
            newConstBias->name = forwardOp->name + "_Grad_ZeroBias";
            newConstBias->type = OpType_Const;
            result.tensorNames.emplace_back(newConstBias->name);
            newConstBias->outputIndexes             = {zeroBiasId};
            newConstBias->main.type                 = OpParameter_Blob;
            newConstBias->main.value                = new BlobT(*originConstOp->main.AsBlob());
            newConstBias->main.AsBlob()->dataFormat = MNN_DATA_FORMAT_NHWC;
            newConstBias->main.AsBlob()->dims = {ALIGN_UP4(forwardOp->main.AsConvolution2D()->common->inputCount)};
            newConstBias->main.AsBlob()->float32s.resize(
                ALIGN_UP4(forwardOp->main.AsConvolution2D()->common->inputCount));
            std::fill(newConstBias->main.AsBlob()->float32s.begin(), newConstBias->main.AsBlob()->float32s.end(), 0);
            result.opLists.emplace_back(std::move(newConstBias));

            // Create Input Grad
            unique_ptr<OpT> newOp(new OpT);
            newOp->name          = forwardOp->name + "_Input_Grad";
            newOp->inputIndexes  = {outputDiff, forwardOp->inputIndexes[1], zeroBiasId};
            newOp->outputIndexes = {gradTensors[0]};
            if (forwardOp->type == OpType_Convolution) {
                newOp->type = OpType_Deconvolution;
            } else if (forwardOp->type == OpType_ConvolutionDepthwise) {
                newOp->type = OpType_DeconvolutionDepthwise;
            }
            newOp->main.type = OpParameter_Convolution2D;
            auto conv2D      = new Convolution2DT;
            conv2D->common.reset(new Convolution2DCommonT(*forwardOp->main.AsConvolution2D()->common));
            auto inputCount             = conv2D->common->inputCount;
            auto outputCount            = conv2D->common->outputCount;
            conv2D->common->inputCount  = outputCount;
            conv2D->common->outputCount = inputCount;
            newOp->main.value           = conv2D;
            result.opLists.emplace_back(std::move(newOp));
        }
        // Add Filter Grad
        {
            unique_ptr<OpT> newOp(new OpT);
            newOp->name          = forwardOp->name + "_Filter_Grad";
            newOp->inputIndexes  = {forwardOp->inputIndexes[1], forwardOp->inputIndexes[0], outputDiff};
            newOp->outputIndexes = {gradTensors[1]};
            newOp->type          = OpType_Conv2DBackPropFilter;
            newOp->main.type     = OpParameter_Convolution2D;
            auto conv2D          = new Convolution2DT;
            conv2D->common.reset(new Convolution2DCommonT(*forwardOp->main.AsConvolution2D()->common));
            newOp->main.value = conv2D;
            result.opLists.emplace_back(std::move(newOp));
        }

        // Add Bias Grad
        {
            unique_ptr<OpT> newOp(new OpT);
            newOp->name          = forwardOp->name + "_Bias_Grad_Convert";
            newOp->outputIndexes = {result.newTensorOffset + 1};
            result.tensorNames.emplace_back(newOp->name);
            newOp->inputIndexes = {outputDiff};
            newOp->type         = OpType_ConvertTensor;
            newOp->main.type    = OpParameter_TensorConvertInfo;
            auto red            = new TensorConvertInfoT;
            red->source         = MNN_DATA_FORMAT_NC4HW4;
            red->dest           = MNN_DATA_FORMAT_NHWC;
            newOp->main.value   = red;
            result.opLists.emplace_back(std::move(newOp));
        }
        {
            unique_ptr<OpT> newOp(new OpT);
            newOp->name          = forwardOp->name + "_Bias_Grad";
            newOp->inputIndexes  = {result.newTensorOffset + 1};
            newOp->outputIndexes = {gradTensors[2]};
            newOp->type          = OpType_Reduction;
            newOp->main.type     = OpParameter_ReductionParam;
            auto red             = new ReductionParamT;
            red->dim             = {0, 1, 2};
            red->keepDims        = false;
            red->dType           = DataType_DT_FLOAT;
            red->operation       = ReductionType_SUM;
            newOp->main.value    = red;
            result.opLists.emplace_back(std::move(newOp));
        }
        return result;
    }
};
class ConvGradCreator : public OpGrad::Creator {
public:
    virtual OpGrad* onCreate(const MNN::OpT* op, const std::vector<MNN::Tensor*>& inputs,
                             const std::vector<MNN::Tensor*>& outputs) const override {
        return new ConvGrad;
    }
};
static const auto gRegister = []() {
    static ConvGradCreator _c;
    OpGrad::insert(OpType_Convolution, &_c);
    OpGrad::insert(OpType_ConvolutionDepthwise, &_c);
    return true;
}();
