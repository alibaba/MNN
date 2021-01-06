//
//  RemoveInverseTensorConverter.cpp
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

class RemoveInverseTensorConverter {
public:
    RemoveInverseTensorConverter();
};

RemoveInverseTensorConverter::RemoveInverseTensorConverter() {
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
        if (convert1_params->source() != convert2_params->dest()) {
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

        auto* identity   = new MNN::ExtraT;
        identity->type   = "Identity";
        identity->engine = "Tensorflow";
        std::unique_ptr<MNN::OpT> identity_op(new MNN::OpT);
        identity_op->name       = expr->name();
        identity_op->type       = OpType_Extra;
        identity_op->main.type  = OpParameter_Extra;
        identity_op->main.value = identity;

        VARP x   = input_expr->inputs().at(0);
        new_expr = Expr::create(identity_op.get(), {x});

        Expr::replace(expr, new_expr);
        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("RemoveInverseTensorConverter", match, fold, PASS_PRIORITY_LOW);
}

static RemoveInverseTensorConverter g_remove_inverse_tensor_convert;

} // namespace Express
} // namespace MNN
