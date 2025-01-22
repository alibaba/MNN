//
//  ConvBiasAdd.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include "MNN_generated.h"

namespace MNN {
namespace Express {
static EXPRP _transformConv3DWithConv2D(EXPRP expr) {
    std::unique_ptr<MNN::OpT> originOp(expr->get()->UnPack());
    auto common = originOp->main.AsConvolution3D()->common.get();
    auto weightPtr = originOp->main.AsConvolution3D()->weight.data();
    auto biasDataPtr = originOp->main.AsConvolution3D()->bias.data();
    auto input = expr->inputs()[0];
    // Im2Col
    auto one = _Unsqueeze(_Scalar<int32_t>(1), {0});
    auto negone = _Unsqueeze(_Scalar<int>(-1), {0});
    auto sx = _Shape(input, true);
    auto kernelD = common->kernels[0];
    auto kernelH = common->kernels[1];
    auto kernelW = common->kernels[2];
    auto inputChannel = common->inputCount;
    auto outputChannel = common->outputCount;
    auto kdv = _Unsqueeze(_Scalar<int>(kernelD), {0});
    auto w = _Slice(sx, _Unsqueeze(_Scalar<int32_t>(4), {0}), one);
    auto h = _Slice(sx, _Unsqueeze(_Scalar<int32_t>(3), {0}), one);
    auto d = _Slice(sx, _Unsqueeze(_Scalar<int32_t>(2), {0}), one);
    auto ic = _Slice(sx, _Unsqueeze(_Scalar<int32_t>(1), {0}), one);
    auto b = _Slice(sx, _Unsqueeze(_Scalar<int32_t>(0), {0}), one);
    auto im2colinput = _Reshape(input, _Concat({one, b*ic, d, w*h}, 0));
    auto im2coloutput = _Im2Col(im2colinput, {1, kernelD}, {1, common->dilates[0]}, {common->pads[0], 0, common->pads[3], 0}, {1, common->strides[0]});
    // Reshape and Unpack
    std::vector<VARP> convInputs;
    {
        // use -1 to compute od
        auto value = _Reshape(im2coloutput, _Concat({b, ic * kdv, negone, h, w}, 0));
        // Merge od and batch
        value = _Transpose(value, {0, 2, 1, 3, 4});
        value = _Reshape(value, _Concat({negone, ic, kdv, h, w}, 0));
        std::unique_ptr<OpT> op(new OpT);
        op->type       = OpType_Unpack;
        auto axisParam = new AxisT;
        axisParam->axis = 2;
        op->main.type = OpParameter_Axis;
        op->main.value = axisParam;
        EXPRP packexpr = Expr::create(std::move(op), {value}, kernelD);
        convInputs.resize(kernelD);
        for (int i=0; i<kernelD; ++i) {
            convInputs[i] = Variable::create(packexpr, i);
        }
    }
    // Make Conv
    std::vector<VARP> convOutputs(kernelD);
    for (int kd=0; kd<convInputs.size(); ++kd) {
        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_Convolution;
        op->main.type = OpParameter_Convolution2D;
        op->main.value = new Convolution2DT;
        auto conv2D = op->main.AsConvolution2D();
        conv2D->common.reset(new Convolution2DCommonT);
        // Copy common
        auto common2d = conv2D->common.get();
        common2d->inputCount = common->inputCount;
        common2d->outputCount = common->outputCount;
        common2d->hasOutputShape = common->hasOutputShape;
        common2d->dilateX = common->dilates[2];
        common2d->dilateY = common->dilates[1];
        common2d->strideX = common->strides[2];
        common2d->strideY = common->strides[1];
        common2d->pads = {common->pads[1], common->pads[2], common->pads[4], common->pads[5]};
        common2d->kernelX = common->kernels[2];
        common2d->kernelY = common->kernels[1];
        common2d->group = common->group;
        common2d->padMode = common->padMode;
        // Split Weight
        int weightGroupSize = inputChannel*outputChannel / common->group;
        conv2D->weight.resize(kernelH * kernelW * weightGroupSize);
        for (int i=0; i<weightGroupSize; ++i) {
            ::memcpy(conv2D->weight.data() + kernelH * kernelW * i, weightPtr + i * kernelD * kernelH * kernelW + kd * kernelH * kernelW, kernelH * kernelW * sizeof(float));
        }
        conv2D->bias.resize(outputChannel);
        ::memset(conv2D->bias.data(), 0, outputChannel * sizeof(float));
        if (kd == kernelD - 1) {
            ::memcpy(conv2D->bias.data(), biasDataPtr, outputChannel * sizeof(float));
        }
        auto convExpr = Expr::create(std::move(op), {convInputs[kd]}, 1);
        convOutputs[kd] = Variable::create(convExpr);
        convOutputs[kd]->setName(expr->name() + "__" + std::to_string(kd));
    }
    VARP output;
    if (kernelD > 1) {
        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_Eltwise;
        op->main.type = OpParameter_Eltwise;
        op->main.value = new EltwiseT;
        op->main.AsEltwise()->type = EltwiseType_SUM;
        auto eltExpr = Expr::create(std::move(op), convOutputs);
        output = Variable::create(eltExpr);
    } else {
        output = convOutputs[0];
    }
    if (common->relu) {
        output = _Relu(output);
    } else if (common->relu6) {
        output = _Relu6(output);
    }

    // Split od and batch
    sx = _Shape(output, true);
    w = _Slice(sx, _Unsqueeze(_Scalar<int32_t>(3), {0}), one);
    h = _Slice(sx, _Unsqueeze(_Scalar<int32_t>(2), {0}), one);
    auto oc = _Unsqueeze(_Scalar<int>(outputChannel), {0});
    output = _Reshape(output, _Concat({b, negone, oc, h, w}, 0));
    output = _Transpose(output, {0, 2, 1, 3, 4});
    output->expr().first->setName(expr->name());
    return output->expr().first;
}

static auto gRegister = []() {
    auto compare = [](EXPRP expr) {
        if (nullptr == expr->get()) {
            return false;
        }
        if (expr->get()->type() != OpType_Convolution3D) {
            return false;
        }
        return expr->get()->type() == OpType_Convolution3D && expr->inputs().size() == 1;
    };
    auto modify = [](EXPRP expr) {
        auto newExpr = _transformConv3DWithConv2D(expr);
        newExpr->setName(expr->name());
        Expr::replace(expr, newExpr);
        return true;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("Convolution3DTurn2D", compare, modify, PASS_PRIORITY_MIDDLE);
    return true;
}();
}
} // namespace MNN
