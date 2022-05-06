//
//  BinaryAddToEltwise.cpp
//  MNNConverter
//
//  Created by MNN on 2020/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN/expr/MathOp.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {

static auto gRegister = []() {
    auto match = [](EXPRP expr) {
        if (nullptr == expr->get()) {
            return false;
        }
        if (expr->get()->type() != OpType_BinaryOp) {
            return false;
        }
        auto binaryOp     = expr->get();
        auto binaryParams = binaryOp->main_as_BinaryOp();
        if (binaryParams->opType() != BinaryOpOperation_ADD) {
            return false;
        }

        auto leftConvertVar  = expr->inputs()[0];
        auto leftConvertExpr = leftConvertVar->expr().first;
        if (leftConvertExpr->get() == nullptr) {
            return false;
        }
        if (leftConvertExpr->get()->type() != OpType_ConvertTensor) {
            return false;
        }

        auto leftMoreConvertExpr = leftConvertExpr->inputs()[0]->expr().first;
        while (!leftMoreConvertExpr->get() && leftMoreConvertExpr->get()->type() == OpType_ConvertTensor) {
            leftConvertExpr     = leftMoreConvertExpr;
            leftMoreConvertExpr = leftConvertExpr->inputs()[0]->expr().first;
        }

        auto leftInt8ToFloatVar  = leftConvertExpr->inputs()[0];
        auto leftInt8ToFloatExpr = leftInt8ToFloatVar->expr().first;
        if (leftInt8ToFloatExpr->get() == nullptr) {
            return false;
        }
        if (leftInt8ToFloatExpr->get()->type() != OpType_Int8ToFloat) {
            return false;
        }

        auto rightConvertVar  = expr->inputs()[1];
        auto rightConvertExpr = rightConvertVar->expr().first;
        if (rightConvertExpr->get() == nullptr) {
            return false;
        }
        if (rightConvertExpr->get()->type() != OpType_ConvertTensor) {
            return false;
        }

        auto rightMoreConvertExpr = rightConvertExpr->inputs()[0]->expr().first;
        while (!rightMoreConvertExpr->get() && rightMoreConvertExpr->get()->type() == OpType_ConvertTensor) {
            rightConvertExpr     = rightMoreConvertExpr;
            rightMoreConvertExpr = rightConvertExpr->inputs()[0]->expr().first;
        }

        auto rightInt8ToFloatVar  = rightConvertExpr->inputs()[0];
        auto rightInt8ToFloatExpr = rightInt8ToFloatVar->expr().first;
        if (rightInt8ToFloatExpr->get() == nullptr) {
            return false;
        }
        if (rightInt8ToFloatExpr->get()->type() != OpType_Int8ToFloat) {
            return false;
        }

        return true;
    };

    auto transform = [](EXPRP expr) {
        auto leftConvertVar  = expr->inputs()[0];
        auto leftConvertExpr = leftConvertVar->expr().first;
        auto leftConvertOp   = leftConvertExpr->get();

        auto leftInt8ToFloatVar  = leftConvertExpr->inputs()[0];
        auto leftInt8ToFloatExpr = leftConvertVar->expr().first;

        auto rightConvertVar  = expr->inputs()[1];
        auto rightConvertExpr = rightConvertVar->expr().first;
        auto rightConvertOp   = rightConvertExpr->get();

        auto rightInt8ToFloatVar  = rightConvertExpr->inputs()[0];
        auto rightInt8ToFloatExpr = rightConvertVar->expr().first;

        auto eltwiseSum = _Sum(leftInt8ToFloatVar, rightInt8ToFloatVar, {});
        eltwiseSum->setName(expr->name());

        Expr::replace(expr, eltwiseSum->expr().first);
        return true;
    };

    TemplateMerge::getInstance("Merge").insertTemplate("BinaryAddToEltwise", match, transform, PASS_PRIORITY_MIDDLE);
    return true;
}();

static auto gRegister2 = []() {
    auto match = [](EXPRP expr) {
        if (nullptr == expr->get()) {
            return false;
        }
        if (expr->get()->type() != OpType_BinaryOp) {
            return false;
        }
        auto binaryOp     = expr->get();
        auto binaryParams = binaryOp->main_as_BinaryOp();
        if (binaryParams->opType() != BinaryOpOperation_POW) {
            return false;
        }

        auto rightInputVar  = expr->inputs()[1];
        if (rightInputVar->getInfo() == nullptr) {
            return false;
        }
        if (rightInputVar->getInfo()->type == halide_type_of<float>()) {
            return false;
        }
        return true;
    };

    auto transform = [](EXPRP expr) {
        auto leftInputVar  = expr->inputs()[0];
        auto rightInputVar  = expr->inputs()[1];
        auto cast = _Cast<float>(rightInputVar);
        cast->setName(expr->name() + "_cast");
        auto newPow = _Pow(leftInputVar, cast);
        Expr::replace(expr, newPow->expr().first);
        return true;
    };

    TemplateMerge::getInstance("Merge").insertTemplate("PowInputCast", match, transform, PASS_PRIORITY_MIDDLE);
    return true;
}();
}
} // namespace MNN
