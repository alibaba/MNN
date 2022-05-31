//
//  RemoveDuplicateReshape.cpp
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

class RemoveDuplicateReshape {
public:
    RemoveDuplicateReshape();
};

RemoveDuplicateReshape::RemoveDuplicateReshape() {
    auto match = [this](EXPRP expr) -> bool {
        if (!expr->get() || expr->get()->type() != OpType_Reshape) {
            return false;
        }
        if (expr->inputs().size() < 1) {
            return false;
        }

        VARP input         = expr->inputs().at(0);
        const Op* input_op = input->expr().first->get();
        if (!input_op || (input_op->type() != OpType_Reshape && input_op->type() != OpType_Squeeze)) {
            return false;
        }
        return true;
    };

    auto fold = [this](EXPRP expr) -> bool {
        VARP input       = expr->inputs().at(0);
        EXPRP input_expr = input->expr().first;
        if (!input_expr->inputs().size()) {
            return false;
        }
        input = input_expr->inputs().at(0);

        auto* param = expr->get()->main_as_Reshape();
        std::vector<int> dims;
        if (param->dims()) {
            for (int i = 0; i < param->dims()->size(); ++i) {
                dims.push_back(param->dims()->Get(i));
            }
        }
        auto* reshape    = new MNN::ReshapeT;
        reshape->dims    = dims;
        reshape->dimType = param->dimType();
        std::unique_ptr<OpT> reshape_op(new OpT);
        reshape_op->name       = expr->name();
        reshape_op->type       = OpType_Reshape;
        reshape_op->main.type  = OpParameter_Reshape;
        reshape_op->main.value = reshape;

        EXPRP reshape_expr;
        if (expr->inputs().size() == 2) {
            reshape_expr = Expr::create(reshape_op.get(), {input, expr->inputs().at(1)}, 1);
        } else {
            reshape_expr = Expr::create(reshape_op.get(), {input}, 1);
        }
        reshape_expr->setName(expr->name());
        Expr::replace(expr, reshape_expr);
        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("RemoveDuplicateReshape", match, fold);
}

//static RemoveDuplicateReshape g_remove_duplicate_reshape;

} // namespace Express
} // namespace MNN
