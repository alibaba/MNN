//
//  FuseTfPrelu.cpp
//  MNNConverter
//
//  Created by MNN on 2020/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN/expr/MathOp.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"
#include "MNN_generated.h"
#include "../../common/Global.hpp"

namespace MNN {
namespace Express {

enum PreluCases {
    None,
    OCRCustom,
};

static auto gRegister = []() {
    auto match = [](EXPRP expr) {
        PreluCases preluCase = PreluCases::None;

        // ocr custom case of prelu
        {
            if (nullptr == expr->get()) {
                return false;
            }
            if (expr->get()->type() != OpType_Eltwise) {
                return false;
            }
            if (expr->get()->main_as_Eltwise()->type() != EltwiseType_SUM) {
                return false;
            }
            if (expr->inputs().size() != 2) {
                return false;
            }

            auto leftReluVar = expr->inputs().at(0);
            auto leftReluExpr = leftReluVar->expr().first;
            if (leftReluExpr->get() == nullptr) {
                return false;
            }
            if (leftReluExpr->get()->type() != OpType_ReLU) {
                return false;
            }

            auto rightBinaryVar = expr->inputs().at(1);
            auto rightBinaryExpr = rightBinaryVar->expr().first;
            if (rightBinaryExpr->get() == nullptr) {
                return false;
            }
            if (rightBinaryExpr->get()->type() != OpType_BinaryOp) {
                return false;
            }
            if (rightBinaryExpr->get()->main_as_BinaryOp()->opType() != BinaryOpOperation_MUL) {
                return false;
            }

            auto rightBinaryConstVar = rightBinaryExpr->inputs().at(0);
            auto rightBinaryConstExpr = rightBinaryConstVar->expr().first;
            if (rightBinaryConstExpr->get() != nullptr) {
                return false;
            }
            auto rightBinaryReluVar = rightBinaryExpr->inputs().at(1);
            auto rightBinaryReluExpr = rightBinaryReluVar->expr().first;
            if (rightBinaryReluExpr->get() == nullptr) {
                return false;
            }
            bool cond = ((rightBinaryConstExpr->inputType() == VARP::CONSTANT) && (rightBinaryReluExpr->get()->type() == OpType_ReLU));
            if (!cond) {
                return false;
            }

            auto unaryVar = rightBinaryReluExpr->inputs().at(0);
            auto unaryExpr = unaryVar->expr().first;
            if (unaryExpr->get() == nullptr) {
                return false;
            }
            if (unaryExpr->get()->type() != OpType_UnaryOp) {
                return false;
            }
            if (unaryExpr->get()->main_as_UnaryOp()->opType() != UnaryOpOperation_NEG) {
                return false;
            }

            auto leftSourceVar = leftReluExpr->inputs().at(0);
            auto rightSourceVar = unaryExpr->inputs().at(0);
            if (leftSourceVar->expr() != rightSourceVar->expr()) {
                return false;
            }

            preluCase = PreluCases::OCRCustom;
        }

        Global<PreluCases>::Reset(&preluCase);

        if (preluCase != PreluCases::None) {
            return true;
        }

        return false;
    };

    auto transform = [](EXPRP expr) {
        auto preluCase = Global<PreluCases>::Get();

        // ocr custom case of prelu
        if (*preluCase == PreluCases::OCRCustom) {
            auto leftReluVar = expr->inputs().at(0);
            auto leftReluExpr = leftReluVar->expr().first;
            auto sourceVar = leftReluExpr->inputs().at(0);
            auto rightBinaryVar = expr->inputs().at(1);
            auto rightBinaryExpr = rightBinaryVar->expr().first;
            auto rightBinaryConstVar = rightBinaryExpr->inputs().at(0);

            std::unique_ptr<MNN::OpT> PreluOp(new OpT);
            PreluOp->type       = OpType_PReLU;
            PreluOp->name       = expr->name();
            PreluOp->main.type  = OpParameter_PRelu;
            PreluOp->main.value = new PReluT;
            auto PreluParameter = PreluOp->main.AsPRelu();
            {
                auto PreluPoint     = _Negative(rightBinaryConstVar);
                auto PreluPointInfo = PreluPoint->getInfo();
                auto PreluPointPtr  = PreluPoint->readMap<float>();
                PreluParameter->slope.resize(PreluPointInfo->size);
                ::memcpy(PreluParameter->slope.data(), PreluPointPtr, PreluPointInfo->size * sizeof(float));
                PreluParameter->slopeCount = PreluPointInfo->size;
            }
            auto newVar = Variable::create(Expr::create(PreluOp.get(), {sourceVar}, expr->outputSize()));
            newVar->setName(expr->outputName(0));
            Expr::replace(expr, newVar->expr().first);

            return true;
        }

        return false;
    };

    TemplateMerge::getInstance("Merge").insertTemplate("FuseTfPrelu", match, transform);
    return true;
}();

}
} // namespace MNN
