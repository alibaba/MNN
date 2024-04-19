//
//  ConstantFolding.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"
#include "MergeHelpers.hpp"
#include "Utils.hpp"

namespace MNN {
namespace Express {

class ConstantFolding {
public:
    ConstantFolding();
};

ConstantFolding::ConstantFolding() {
    auto match = [](EXPRP expr) -> bool {
        if (!expr->get()) {
            return false;
        }
        // There's no nodes to be fold if it has no inputs.
        if (!expr->inputs().size()) {
            return false;
        }
        for (const VARP& input : expr->inputs()) {
            const Op* input_op = input->expr().first->get();
            if ((input_op && input_op->type() != OpType_Const) ||
                (!input_op && input->expr().first->inputType() != VARP::CONSTANT)) {
                return false;
            }
        }
        return true /*matched*/;
    };

    auto fold = [this](EXPRP expr) -> bool {
        std::vector<VARP> outputs = helpers::OutputVars(expr);
        if (outputs.size() == 1) {
            VARP output;
            if (outputs.size() == 0) {
                output = Variable::create(expr);
            } else {
                output = outputs.at(0);
            }
            auto output_info = output->getInfo();
            if (!output_info) {
                return false;
            }
            const void* output_data = output->readMap<void>();
            VARP const_var          = _Const(output_data, output_info->dim, output_info->order, output_info->type);
            const_var->setName(expr->name());
            EXPRP constant = const_var->expr().first;
            constant->setName(expr->name());
            expr->inside()->mCache = nullptr;
            Expr::replace(expr, constant);
        } else {
            // TODO(): Support multiple outputs.
            return false;
        }
        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("ConstantFolding", match, fold);
}

static ConstantFolding g_constant_folding;

} // namespace Express
} // namespace MNN
