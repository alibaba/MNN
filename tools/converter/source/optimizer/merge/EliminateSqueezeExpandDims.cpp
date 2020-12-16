//
//  EliminateSqueezeExpandDims.cpp
//  MNNConverter
//
//  Created by MNN on 2020/12/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {

class EliminateSqueezeExpandDims {
public:
    EliminateSqueezeExpandDims();
};

EliminateSqueezeExpandDims::EliminateSqueezeExpandDims() {
    auto match = [this](EXPRP expr) -> bool {
        if (!expr->get()) {
            return false;
        }
        if ((expr->get()->type() != OpType_Squeeze) && (expr->get()->type() != OpType_ExpandDims)) {
            return false;
        }

        VARP input = expr->inputs().at(0);
        const Op* inputOp = input->expr().first->get();
        if (inputOp == nullptr) {
            return false;
        }

        if (input->expr().first->outputSize() != 1) {
            return false;
        }

        if (expr->get()->type() == OpType_Squeeze) {
            if (inputOp->type() != OpType_ExpandDims) {
                return false;
            }
            auto squeezeDims = expr->get()->main_as_SqueezeParam()->squeezeDims();
            int expandDim = inputOp->main_as_ExpandDims()->axis();
            if (squeezeDims->size() != 1) { // squeeze can apply to multi-dimension, but expand_dims can only have single axis value
                return false;
            }
            if (expandDim != squeezeDims->data()[0]) {
                return false;
            }
        }

        if (expr->get()->type() == OpType_ExpandDims) {
            if (inputOp->type() != OpType_Squeeze) {
                return false;
            }
            auto squeezeDims = inputOp->main_as_SqueezeParam()->squeezeDims();
            int expandDim = expr->get()->main_as_ExpandDims()->axis();
            if (squeezeDims->size() != 1) { // squeeze can apply to multi-dimension, but expand_dims can only have single axis value
                return false;
            }
            if (expandDim != squeezeDims->data()[0]) {
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

    TemplateMerge::getInstance("Merge").insertTemplate("EliminateSqueezeExpandDims", match, fold);
}

static EliminateSqueezeExpandDims g_eliminate_squeeze_expand_dims;

} // namespace Express
} // namespace MNN
