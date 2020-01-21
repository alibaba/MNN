//
//  ConvGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN::Express;
using namespace MNN;

class ConvGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto inputs = expr->inputs();
        if (inputs.size() == 1) {
            return std::vector<Express::VARP>{nullptr};
        }
        std::vector<VARP> res(inputs.size(), nullptr);
        auto forwardName = expr->name();
        std::shared_ptr<OpT> forwardOp(expr->get()->UnPack());
        auto outputDiff = backwardOutput[0];
        {
            // Create Input Grad
            unique_ptr<OpT> newOp(new OpT);
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

            auto expr = Expr::create(std::move(newOp), {outputDiff, inputs[1]});
            res[0]    = Variable::create(expr);
            res[0]->setName(forwardName + "_Input_Grad");
        }
        // Add Filter Grad
        {
            unique_ptr<OpT> newOp(new OpT);
            newOp->type      = OpType_Conv2DBackPropFilter;
            newOp->main.type = OpParameter_Convolution2D;
            auto conv2D      = new Convolution2DT;
            conv2D->common.reset(new Convolution2DCommonT(*forwardOp->main.AsConvolution2D()->common));
            newOp->main.value = conv2D;
            auto expr         = Expr::create(std::move(newOp), {inputs[1], inputs[0], outputDiff});
            res[1]            = Variable::create(expr);
            res[1]->setName(forwardName + "_Filter_Grad");
        }
        // Add Bias Grad
        if (inputs.size() > 2) {
            auto gradConvert = _Convert(outputDiff, NHWC);
            res[2]           = _ReduceSum(gradConvert, {0, 1, 2});
            res[2]->setName(forwardName + "_Bias_Grad");
        }
        return res;
    }
};

class DeconvGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto inputs = expr->inputs();
        if (inputs.size() == 1) {
            return std::vector<Express::VARP>{nullptr};
        }
        std::vector<VARP> res(inputs.size(), nullptr);
        auto forwardName = expr->name();
        std::shared_ptr<OpT> forwardOp(expr->get()->UnPack());
        auto outputDiff = backwardOutput[0];
        {
            // Create Input Grad
            unique_ptr<OpT> newOp(new OpT);
            if (forwardOp->type == OpType_Deconvolution) {
                newOp->type = OpType_Convolution;
            } else if (forwardOp->type == OpType_DeconvolutionDepthwise) {
                newOp->type = OpType_ConvolutionDepthwise;
            }
            newOp->main.type = OpParameter_Convolution2D;
            auto conv2D      = new Convolution2DT;
            conv2D->common.reset(new Convolution2DCommonT(*forwardOp->main.AsConvolution2D()->common));
            auto inputCount             = conv2D->common->inputCount;
            auto outputCount            = conv2D->common->outputCount;
            conv2D->common->inputCount  = outputCount;
            conv2D->common->outputCount = inputCount;
            newOp->main.value           = conv2D;

            auto expr = Expr::create(std::move(newOp), {outputDiff, inputs[1]});
            res[0]    = Variable::create(expr);
            res[0]->setName(forwardName + "_Input_Grad");
        }
        // Add Filter Grad
        {
            unique_ptr<OpT> newOp(new OpT);
            newOp->type      = OpType_Conv2DBackPropFilter;
            newOp->main.type = OpParameter_Convolution2D;
            auto conv2D      = new Convolution2DT;
            conv2D->common.reset(new Convolution2DCommonT(*forwardOp->main.AsConvolution2D()->common));
            newOp->main.value = conv2D;
            auto expr         = Expr::create(std::move(newOp), {inputs[1], outputDiff, inputs[0]});
            res[1]            = Variable::create(expr);
            res[1]->setName(forwardName + "_Filter_Grad");
        }
        // Add Bias Grad
        if (inputs.size() > 2) {
            auto gradConvert = _Convert(outputDiff, NHWC);
            res[2]           = _ReduceSum(gradConvert, {0, 1, 2});
            res[2]->setName(forwardName + "_Bias_Grad");
        }
        return res;
    }
};

static const auto gRegister = []() {
    static ConvGrad _c;
    OpGrad::insert(OpType_Convolution, &_c);
    OpGrad::insert(OpType_ConvolutionDepthwise, &_c);
    static DeconvGrad _d;
    OpGrad::insert(OpType_Deconvolution, &_d);
    OpGrad::insert(OpType_DeconvolutionDepthwise, &_d);
    return true;
}();
