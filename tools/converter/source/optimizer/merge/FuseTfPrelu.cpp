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

namespace MNN {
namespace Express {

enum PreluCases {
    None,
    OCRCustom,
};

auto getPreluCases = [](EXPRP expr) {
    auto NotPrelu = PreluCases::None;
    // ocr custom case of prelu
    {
        if (nullptr == expr->get()) {
            return NotPrelu;
        }
        if (expr->get()->type() != OpType_Eltwise) {
            return NotPrelu;
        }
        if (expr->get()->main_as_Eltwise()->type() != EltwiseType_SUM) {
            return NotPrelu;
        }
        if (expr->inputs().size() != 2) {
            return NotPrelu;
        }

        auto leftReluVar = expr->inputs().at(0);
        auto leftReluExpr = leftReluVar->expr().first;
        if (leftReluExpr->get() == nullptr) {
            return NotPrelu;
        }
        if (leftReluExpr->get()->type() != OpType_ReLU) {
            return NotPrelu;
        }

        auto rightBinaryVar = expr->inputs().at(1);
        auto rightBinaryExpr = rightBinaryVar->expr().first;
        if (rightBinaryExpr->get() == nullptr) {
            return NotPrelu;
        }
        if (rightBinaryExpr->get()->type() != OpType_BinaryOp) {
            return NotPrelu;
        }
        if (rightBinaryExpr->get()->main_as_BinaryOp()->opType() != BinaryOpOperation_MUL) {
            return NotPrelu;
        }

        auto rightBinaryConstVar = rightBinaryExpr->inputs().at(0);
        auto rightBinaryConstExpr = rightBinaryConstVar->expr().first;
        if (rightBinaryConstExpr->get() != nullptr) {
            return NotPrelu;
        }
        auto rightBinaryReluVar = rightBinaryExpr->inputs().at(1);
        auto rightBinaryReluExpr = rightBinaryReluVar->expr().first;
        if (rightBinaryReluExpr->get() == nullptr) {
            return NotPrelu;
        }
        bool cond = ((rightBinaryConstExpr->inputType() == VARP::CONSTANT) && (rightBinaryReluExpr->get()->type() == OpType_ReLU));
        if (!cond) {
            return NotPrelu;
        }

        auto unaryVar = rightBinaryReluExpr->inputs().at(0);
        auto unaryExpr = unaryVar->expr().first;
        if (unaryExpr->get() == nullptr) {
            return NotPrelu;
        }
        if (unaryExpr->get()->type() != OpType_UnaryOp) {
            return NotPrelu;
        }
        if (unaryExpr->get()->main_as_UnaryOp()->opType() != UnaryOpOperation_NEG) {
            return NotPrelu;
        }

        auto leftSourceVar = leftReluExpr->inputs().at(0);
        auto rightSourceVar = unaryExpr->inputs().at(0);
        if (leftSourceVar->expr() != rightSourceVar->expr()) {
            return NotPrelu;
        }

        return PreluCases::OCRCustom;
    }

    return NotPrelu;
};

static auto gRegister = []() {
    auto match = [](EXPRP expr) {
        auto preluCase = getPreluCases(expr);

        if (preluCase != PreluCases::None) {
            return true;
        }

        return false;
    };

    auto transform = [](EXPRP expr) {
        auto preluCase = getPreluCases(expr);

        // ocr custom case of prelu
        if (preluCase == PreluCases::OCRCustom) {
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
