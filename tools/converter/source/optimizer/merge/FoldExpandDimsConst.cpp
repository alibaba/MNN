//
//  FoldExpandDimsConst.cpp
//  MNNConverter
//
//  Created by MNN on 2020/12/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {

class FoldExpandDimsConst {
public:
    FoldExpandDimsConst();
};

FoldExpandDimsConst::FoldExpandDimsConst() {
    auto match = [this](EXPRP expr) -> bool {
         // Check the current op is `ExpandDims` op.
         if (!expr->get() || expr->get()->type() != OpType_ExpandDims ||
             expr->inputs().size() < 2) {
             return false;
         }
         // Check the second input is `Constant` op.
         VARP axis = expr->inputs().at(1);
         if (axis->expr().first->inputType() != VARP::CONSTANT) {
             return false;
         }
         return true;
    };

    auto fold = [this](EXPRP expr) -> bool {
        VARP axis = expr->inputs().at(1);
         int axis_val = axis->readMap<int>()[0];
         std::unique_ptr<OpT> expand_dims_op(expr->get()->UnPack());
         expand_dims_op->main.AsExpandDims()->axis = axis_val;

         auto expand_dims = Expr::create(expand_dims_op.get(),  // NOLINT
                                         {expr->inputs().at(0)});
         Expr::replace(expr, expand_dims);
         return true;
    };

    TemplateMerge::getInstance("Merge").insertTemplate("FoldExpandDimsConst", match, fold);
}

static FoldExpandDimsConst g_fold_expand_dims_const;

} // namespace Express
} // namespace MNN
