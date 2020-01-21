//
//  ConvBiasAdd.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN_generated.h"


namespace MNN {
namespace Express {
static auto gRegister = []() {
    auto compare = [](EXPRP expr) {
        if (nullptr == expr->get()) {
            return false;
        }
        if (expr->get()->type() != OpType_BinaryOp) {
            return false;
        }
        if (expr->get()->main_as_BinaryOp()->opType() != BinaryOpOperation_ADD) {
            return false;
        }
        auto inputs = expr->inputs();
        auto inputExpr = inputs[0]->expr().first;
        if (nullptr == inputExpr->get()) {
            return false;
        }
        if (inputExpr->get()->main_type() != OpParameter_Convolution2D || inputExpr->outputs().size() != 1) {
            return false;
        }
        // Merge into convolution
        auto biasVar = inputs[1];
        auto biasInfo = biasVar->getInfo();
        auto biasPtr = biasVar->readMap<float>();
        if (nullptr == biasInfo || nullptr == biasPtr) {
            return false;;
        }
        auto paraent = inputExpr->inputs();
        auto outputCount = inputExpr->get()->main_as_Convolution2D()->common()->outputCount();
        if (biasInfo->size != outputCount) {
            return false;
        }
        return true;
    };
    auto modify = [](EXPRP expr) {
        auto inputs = expr->inputs();
        auto inputExpr = inputs[0]->expr().first;
        auto biasVar = inputs[1];
        auto biasInfo = biasVar->getInfo();
        auto biasPtr = biasVar->readMap<float>();
        std::unique_ptr<OpT> convOp(inputExpr->get()->UnPack());
        auto& biasData = convOp->main.AsConvolution2D()->bias;
        MNN_ASSERT(biasInfo->size == biasData.size());
        for (int i=0; i<biasData.size(); ++i) {
            biasData[i] += biasPtr[i];
        }
        auto newExpr = Expr::create(convOp.get(), inputExpr->inputs());
        newExpr->setName(expr->name());
        Expr::replace(expr, newExpr);
        return true;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("ConvBiasAdd", compare, modify);
    return true;
}();
}
} // namespace MNN
