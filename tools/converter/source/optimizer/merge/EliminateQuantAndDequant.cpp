//
//  EliminateQuantAndDequant.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {

class EliminateQuantAndDequant {
public:
    EliminateQuantAndDequant();
};

EliminateQuantAndDequant::EliminateQuantAndDequant() {
    auto match = [this](EXPRP expr) -> bool {
        if (!expr->get() || (expr->get()->type() != OpType_FloatToInt8 && expr->get()->type() != OpType_Int8ToFloat)) {
            return false;
        }
        VARP input         = expr->inputs().at(0);
        const Op* input_op = input->expr().first->get();
        if (!input_op) {
            return false;
        }
        if (expr->get()->type() == OpType_FloatToInt8) {
            if (input_op->type() != OpType_Int8ToFloat) {
                return false;
            }
        }
        if (expr->get()->type() == OpType_Int8ToFloat) {
            if (input_op->type() != OpType_FloatToInt8) {
                return false;
            }
        }
        return true;
    };

    auto fold = [this](EXPRP expr) -> bool {
        VARP input = expr->inputs().at(0);
        input      = input->expr().first->inputs().at(0);

        auto* identity   = new MNN::ExtraT;
        identity->type   = "Identity";
        identity->engine = "Tensorflow";
        std::unique_ptr<MNN::OpT> identity_op(new MNN::OpT);
        identity_op->name       = expr->name();
        identity_op->type       = OpType_Extra;
        identity_op->main.type  = OpParameter_Extra;
        identity_op->main.value = identity;

        EXPRP identity_expr = Expr::create(identity_op.get(), {input});
        Expr::replace(expr, identity_expr);
        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("EliminateQuantAndDequant", match, fold, PASS_PRIORITY_LOW);
}

static EliminateQuantAndDequant g_eliminate_quant_dequant;

} // namespace Express
} // namespace MNN
