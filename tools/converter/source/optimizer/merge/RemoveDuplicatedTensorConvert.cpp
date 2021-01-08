//
//  RemoveDuplicatedTensorConvert.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"
#include "MergeHelpers.hpp"

namespace MNN {
namespace Express {

class RemoveDuplicatedTensorConvert {
public:
    RemoveDuplicatedTensorConvert();
};

RemoveDuplicatedTensorConvert::RemoveDuplicatedTensorConvert() {
    auto match = [this](EXPRP expr) -> bool {
        if (!expr->get() || expr->get()->type() != OpType_ConvertTensor) {
            return false;
        }
        VARP input       = expr->inputs().at(0);
        EXPRP input_expr = input->expr().first;
        if (!input_expr->get() || input_expr->get()->type() != OpType_ConvertTensor) {
            return false;
        }

        const auto* convert1_params = input_expr->get()->main_as_TensorConvertInfo();
        const auto* convert2_params = expr->get()->main_as_TensorConvertInfo();
        if (convert1_params->source() != convert2_params->source()) {
            return false;
        }
        if (convert1_params->dest() != convert2_params->dest()) {
            return false;
        }

        return true;
    };

    auto fold = [this](EXPRP expr) -> bool {
        VARP input      = expr->inputs().at(0);
        auto input_expr = input->expr().first;

        const auto* convert1_params = input_expr->get()->main_as_TensorConvertInfo();
        const auto* convert2_params = expr->get()->main_as_TensorConvertInfo();
        EXPRP new_expr;
        
        VARP x   = input_expr->inputs().at(0);
        new_expr = Expr::create(expr->extra(), {x});
        new_expr->setName(expr->name());

        Expr::replace(expr, new_expr);
        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("RemoveDuplicatedTensorConvert", match, fold, PASS_PRIORITY_LOW);
}

static RemoveDuplicatedTensorConvert g_remove_duplicated_tensor_convert;

} // namespace Express
} // namespace MNN
