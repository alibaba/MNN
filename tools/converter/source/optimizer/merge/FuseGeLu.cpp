//
//  FuseGeLu.cpp
//  MNNConverter
//
//  Created by MNN on 2021/04/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <unordered_map>

#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"
#include "MergeHelpers.hpp"

namespace MNN {
namespace Express {

class FuseGeLu {
public:
    FuseGeLu();
private:
    VARP gelu_inputs_;
};

bool IsBinaryPow(EXPRP expr) {
    // return true;
    const Op* op = expr->get();
    if (op == nullptr || op->type() != OpType_BinaryOp) {
        return false;
    }
    return op->main_as_BinaryOp()->opType() == BinaryOpOperation_POW;
}

bool IsTanh(EXPRP expr) {
    const Op* op = expr->get();
    if (op == nullptr) {
        return false;
    }
    if (op->type() == OpType_TanH) {
        return true;
    }
    if (op->type() == OpType_UnaryOp && op->main_as_UnaryOp()->opType() == UnaryOpOperation_TANH) {
        return true;
    }
    return false;
}


static bool scalarEquals(VARP v, float expected) {
    if (!helpers::IsConstant(v->expr().first)) {
        return false;
    }
    const auto info = v->getInfo();
    if (!info || info->size != 1 || info->type.code != halide_type_float || info->type.bits != 32) {
        return false;
    }
    auto value = v->readMap<float>();
    return value != nullptr && *value == expected;
}

FuseGeLu::FuseGeLu() {
    auto match = [this](EXPRP expr) -> bool {
        if (!expr->get() || !helpers::IsBinaryMul(expr)) {
            return false;
        }
        auto select = [](EXPRP op, float coefficient) -> VARP {
            if (scalarEquals(op->inputs().at(0), coefficient)) {
                return op->inputs().at(1);
            }
            if (scalarEquals(op->inputs().at(1), coefficient)) {
                return op->inputs().at(0);
            }
            return nullptr;
        };
        for (int i = 0; i < 2; ++i) {
            auto outer = expr->inputs().at(i);
            auto half = expr->inputs().at(1 - i)->expr().first;
            if (!helpers::IsBinaryMul(half)) continue;
            auto sum = select(half, 0.5f);
            if (!sum.get() || !helpers::IsBinaryAdd(sum->expr().first)) continue;
            auto tanh = select(sum->expr().first, 1.0f);
            if (!tanh.get() || !IsTanh(tanh->expr().first)) continue;
            auto scaled = tanh->expr().first->inputs().at(0)->expr().first;
            if (!helpers::IsBinaryMul(scaled)) continue;
            auto polynomial = select(scaled, 0.7978845608028654f);
            if (!polynomial.get() || !helpers::IsBinaryAdd(polynomial->expr().first)) continue;
            auto terms = polynomial->expr().first->inputs();
            for (int j = 0; j < 2; ++j) {
                if (terms.at(j).get() != outer.get()) continue;
                auto term = terms.at(1 - j)->expr().first;
                if (!helpers::IsBinaryMul(term)) continue;
                auto cube = select(term, 0.044715f);
                if (!cube.get() || !IsBinaryPow(cube->expr().first)) continue;
                auto args = cube->expr().first->inputs();
                if (args.at(0).get() != outer.get() || !scalarEquals(args.at(1), 3.f)) continue;
                gelu_inputs_ = outer;
                return true;
            }
        }
        return false;
    };

    auto fold = [this](EXPRP expr) -> bool {
        auto config = Global<modelConfig>::Get();
        auto version = config->targetVersion;
        if (version < 1.2f) {
            // For target version < 1.2 , don't support gelu
            return false;
        }
        std::unique_ptr<OpT> gelu_op(new OpT);
        gelu_op->name       = expr->name();
        gelu_op->type       = OpType_UnaryOp;
        gelu_op->main.type  = OpParameter_UnaryOp;
        gelu_op->main.value = new UnaryOpT;
        auto geluParam = gelu_op->main.AsUnaryOp();
        geluParam->opType = UnaryOpOperation_GELU;
        auto gelu_expr = Variable::create(Expr::create(gelu_op.get(), { gelu_inputs_ }, 1));
        gelu_expr->setName(expr->name());
        Expr::replace(expr, gelu_expr->expr().first);
        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("FuseGeLu", match, fold);
}

static FuseGeLu g_fuse_gelu;

} // namespace Express
} // namespace MNN
