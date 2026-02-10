//
//  ConvPad.cpp
//  MNNConverter
//
//  Created by MNN on 2026/01/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cmath>
#include "../TemplateMerge.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {
static auto gRegister = []() {
    auto compare = [](EXPRP expr) {
        if (nullptr == expr->get()) {
            return false;
        }
        if (expr->get()->type() != OpType_Convolution && expr->get()->type() != OpType_ConvolutionDepthwise) {
            return false;
        }
        auto common = expr->get()->main_as_Convolution2D()->common();
        if (common->padMode() != PadMode_CAFFE) {
            return false;
        }
        auto inputs    = expr->inputs();
        auto inputExpr = inputs[0]->expr().first;
        if (nullptr == inputExpr->get()) {
            return false;
        }
        if (inputExpr->get()->type() != OpType_Padding) {
            return false;
        }
        auto padParam = inputExpr->get()->main_as_PadParam();
        if (padParam && padParam->mode() != PadValueMode_CONSTANT) {
            return false;
        }
        auto padExpr = inputExpr;
        auto pads = padExpr->inputs()[1];
        bool padZero = true;
        if (padExpr->inputs().size() >= 3) {
            auto padValue = padExpr->inputs()[2]->readMap<float>();
            if (nullptr == padValue) {
                return false;
            }
            padZero = fabsf(padValue[0]) <= 0.00000001f;
        }
        if (!padZero) {
            return false;
        }
        if (pads->readMap<int>() == nullptr || pads->getInfo()->size != 8) {
            return false;
        }
        for (int i=0; i<4; ++i) {
            if (pads->readMap<int>()[i] != 0) {
                // Don't support pad batch and channel
                return false;
            }
        }
        return true;
    };
    auto modify = [](EXPRP expr) {
        auto inputs    = expr->inputs();
        auto pads = inputs[0]->expr().first->inputs()[1]->readMap<int>();
        inputs[0] = inputs[0]->expr().first->inputs()[0];
        std::unique_ptr<OpT> convOp(expr->get()->UnPack());
        auto common = convOp->main.AsConvolution2D()->common.get();
        if (common->pads.empty()) {
            common->pads = {common->padY, common->padX, common->padY, common->padX};
        }
        common->pads[0] += pads[4];
        common->pads[2] += pads[5];
        common->pads[1] += pads[6];
        common->pads[3] += pads[7];
        auto newExpr = Expr::create(convOp.get(), inputs);
        newExpr->setName(expr->name());
        Expr::replace(expr, newExpr);
        return true;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("ConvPad", compare, modify);
    return true;
}();
}
} // namespace MNN
