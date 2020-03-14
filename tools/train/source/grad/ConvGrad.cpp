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
        //FUNC_PRINT_ALL(_ReduceMax(outputDiff)->readMap<float>()[0], f);
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
            auto padMode = conv2D->common->padMode;
            if ((conv2D->common->strideX > 1 || conv2D->common->strideY > 1)) {
                auto inputShape = inputs[0]->getInfo();
                auto outputShape = outputDiff->getInfo();
                if (nullptr == inputShape || nullptr == outputShape) {
                    return {};
                }
                auto iw = inputShape->dim[3];
                auto ih = inputShape->dim[2];
                auto ow = outputShape->dim[3];
                auto oh = outputShape->dim[2];
                auto kW = conv2D->common->kernelX;
                auto kH = conv2D->common->kernelY;
                auto sW = conv2D->common->strideX;
                auto sH = conv2D->common->strideY;
                auto dW = conv2D->common->dilateX;
                auto dH = conv2D->common->dilateY;

                std::vector<int> padding {0, 0, 0, 0};
                int kernelWidthSize = dW * (kW - 1) + 1;
                int kernelHeightSize = dH * (kH - 1) + 1;
                int padNeededWidth  = (ow - 1) * sW + kernelWidthSize - iw;
                int padNeededHeight = (oh - 1) * sH + kernelHeightSize - ih;
                if (padMode == PadMode_SAME) {
                    padding[0] = padNeededHeight / 2;
                    padding[1] = padNeededWidth / 2;
                } else if (padMode == PadMode_CAFFE) {
                    if (conv2D->common->pads.empty()) {
                        padding[0] = conv2D->common->padY;
                        padding[1] = conv2D->common->padX;
                    } else {
                        padding[0] = conv2D->common->pads[0];
                        padding[1] = conv2D->common->pads[1];
                    }
                }
                padding[2] = padNeededHeight - padding[0];
                padding[3] = padNeededWidth - padding[1];
                conv2D->common->pads = padding;
                conv2D->common->padMode = PadMode_CAFFE;
            }
            conv2D->common->inputCount  = outputCount;
            conv2D->common->outputCount = inputCount;
            newOp->main.value           = conv2D;

            auto expr = Expr::create(std::move(newOp), {outputDiff, inputs[1]});
            res[0]    = Variable::create(expr);
            auto resultShape = res[0]->getInfo();
            auto inputShape= inputs[0]->getInfo();
            MNN_ASSERT(resultShape->dim[3] == inputShape->dim[3]);
            MNN_ASSERT(resultShape->dim[2] == inputShape->dim[2]);
        }
        // Add Filter Grad
        {
            unique_ptr<OpT> newOp(new OpT);
            newOp->type      = OpType_Conv2DBackPropFilter;
            newOp->main.type = OpParameter_Convolution2D;
            auto conv2D      = new Convolution2DT;
            conv2D->common.reset(new Convolution2DCommonT(*forwardOp->main.AsConvolution2D()->common));
            newOp->main.value = conv2D;
            auto expr         = Expr::create(std::move(newOp), {inputs[0], outputDiff});
            res[1]            = Variable::create(expr);
        }
        // Add Bias Grad
        if (inputs.size() > 2) {
            auto gradConvert = _Convert(outputDiff, NCHW);
            res[2]           = _ReduceSum(gradConvert, {0, 2, 3});
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
        }
        // Add Filter Grad
        {
            unique_ptr<OpT> newOp(new OpT);
            newOp->type      = OpType_Conv2DBackPropFilter;
            newOp->main.type = OpParameter_Convolution2D;
            auto conv2D      = new Convolution2DT;
            conv2D->common.reset(new Convolution2DCommonT(*forwardOp->main.AsConvolution2D()->common));
            newOp->main.value = conv2D;
            // Revert outputdiff and inputs[0] for deconvolution
            auto expr         = Expr::create(std::move(newOp), {outputDiff, inputs[0]});
            res[1]            = Variable::create(expr);
        }
        // Add Bias Grad
        if (inputs.size() > 2) {
            auto gradConvert = _Convert(outputDiff, NCHW);
            res[2]           = _ReduceSum(gradConvert, {0, 2, 3});
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
