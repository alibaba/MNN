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

FuseGeLu::FuseGeLu() {
    auto match = [this](EXPRP expr) -> bool {
        if (!expr->get() || !helpers::IsBinaryMul(expr)) {
            return false;
        }
        EXPRP x, y, z;
        EXPRP add_1, add_2;
        EXPRP mul_1, mul_2, mul_3;
        EXPRP pow, tanh;
        // x * a
        x = expr->inputs().at(0)->expr().first;
        y = expr->inputs().at(1)->expr().first;
        if (helpers::IsBinaryMul(x)) {
            z = x;
        } else if (helpers::IsBinaryMul(y)) {
            z = y;
        } else {
            return false;
        }
        // x + a
        x = z->inputs().at(0)->expr().first;
        y = z->inputs().at(1)->expr().first;
        if (helpers::IsBinaryAdd(x) && helpers::IsConstant(y)){
            z = x;
        } else if (helpers::IsConstant(x) && helpers::IsBinaryAdd(y)){
            z = y;
        } else {
            return false;
        }
        // tan(x) + a
        x = z->inputs().at(0)->expr().first;
        y = z->inputs().at(1)->expr().first;
        if (helpers::IsConstant(x) && IsTanh(y)) {
            z = y;
        } else if (IsTanh(x) && helpers::IsConstant(y)) {
            z = x;
        } else {
            return false;
        }
        // z = x * a
        z = z->inputs().at(0)->expr().first;
        if (!helpers::IsBinaryMul(z)) {
            return false;
        }
        // z = x + y
        x = z->inputs().at(0)->expr().first;
        y = z->inputs().at(1)->expr().first;
        if (helpers::IsBinaryAdd(x) && helpers::IsConstant(y)){
            z = x;
        } else if (helpers::IsConstant(x) && helpers::IsBinaryAdd(y)) {
            z = y;
        } else {
            return false;
        }
        // z = x * a
        x = z->inputs().at(0)->expr().first;
        y = z->inputs().at(1)->expr().first;
        if (helpers::IsBinaryMul(x)) {
            gelu_inputs_ = z->inputs().at(1);
            z = x;
        } else if (helpers::IsBinaryMul(y)) {
            gelu_inputs_ = z->inputs().at(0);
            z = y;
        } else {
            return false;
        }
        // z = x ^ a
        x = z->inputs().at(0)->expr().first;
        y = z->inputs().at(1)->expr().first;
        if (IsBinaryPow(x) && helpers::IsConstant(y)) {
            z = x;
        } else if (helpers::IsConstant(x) && IsBinaryPow(y)) {
            z = y;
        } else {
            return false;
        }
        return true;
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
