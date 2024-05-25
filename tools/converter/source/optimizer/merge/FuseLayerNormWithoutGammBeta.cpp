//
//  FuseLayerNormWithoutGammaBeta.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <unordered_map>

#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"
#include "MergeHelpers.hpp"

namespace MNN {
namespace Express {

static bool loadAxisFromReduction(EXPRP mean_3, std::vector<int>& axis_var_) {
    if (mean_3->inputs().size() > 1) {
        EXPRP axis     = mean_3->inputs().at(1)->expr().first;
        auto axis_var = mean_3->inputs().at(1);
        if (!helpers::IsConstant(axis)) {
            return false;
        }
        auto info = axis_var->getInfo();
        auto dim = axis_var->readMap<int>();
        axis_var_.resize(info->size);
        ::memcpy(axis_var_.data(), dim, info->size * sizeof(int));
    } else {
        auto reduc = mean_3->get()->main_as_ReductionParam();
        if (nullptr == reduc) {
            return false;
        }
        if (reduc->dim() == nullptr) {
            return false;
        }
        axis_var_.resize(reduc->dim()->size());
        ::memcpy(axis_var_.data(), reduc->dim()->data(), reduc->dim()->size() * sizeof(int));
    }
    return true;
}

class FuseLayerNormV4 {
public:
    FuseLayerNormV4();

private:
    std::vector<int> axis_var_;
    VARP x_var_;
    VARP epsilon_var_;
};

FuseLayerNormV4::FuseLayerNormV4() {
    auto match = [this](EXPRP expr) -> bool {
        if (!expr->get() || !helpers::IsBinaryMul(expr)) {
            return false;
        }
        EXPRP sub_2 = expr->inputs().at(0)->expr().first;
        EXPRP rsqrt = expr->inputs().at(1)->expr().first;
        if (!helpers::IsUnaryRsqrt(rsqrt) || !helpers::IsBinarySub(sub_2)) {
            return false;
        }
        EXPRP add_2 = rsqrt->inputs().at(0)->expr().first;
        if (!helpers::IsBinaryAdd(add_2)) {
            return false;
        }
        EXPRP mean_3  = add_2->inputs().at(0)->expr().first;
        EXPRP epsilon = add_2->inputs().at(1)->expr().first;
        if (!helpers::IsReductionMean(mean_3) || !helpers::IsConstant(epsilon)) {
            return false;
        }
        EXPRP square_1 = mean_3->inputs().at(0)->expr().first;
        if (!helpers::IsUnarySquare(square_1)) {
            return false;
        }
        auto axisLoad = loadAxisFromReduction(mean_3, axis_var_);
        if (!axisLoad) {
            return false;
        }
        VARP sub_2_var = expr->inputs().at(0);
        if (square_1->inputs().at(0).get() != sub_2_var.get()) {
            return false;
        }
        EXPRP x      = sub_2->inputs().at(0)->expr().first;
        EXPRP mean_2 = sub_2->inputs().at(1)->expr().first;
        if (!helpers::IsReductionMean(mean_2)) {
            return false;
        }

        VARP x_var    = sub_2->inputs().at(0);
        if (mean_2->inputs().at(0).get() != x_var.get()) {
            return false;
        }
        std::vector<int> axisV2;
        axisLoad = loadAxisFromReduction(mean_2, axisV2);
        if (!axisLoad) {
            return false;
        }
        if (axisV2 != axis_var_) {
            return false;
        }
        x_var_       = x_var;
        epsilon_var_ = add_2->inputs().at(1);
        return true;
    };

    auto fold = [this](EXPRP expr) -> bool {
        std::unique_ptr<MNN::LayerNormT> layer_norm(new MNN::LayerNormT);
        layer_norm->axis = axis_var_;
        layer_norm->epsilon = epsilon_var_->readMap<float>()[0];

        std::unique_ptr<OpT> layer_norm_op(new OpT);
        layer_norm_op->name       = expr->name();
        layer_norm_op->type       = OpType_LayerNorm;
        layer_norm_op->main.type  = OpParameter_LayerNorm;
        layer_norm_op->main.value = layer_norm.release();

        EXPRP layer_norm_expr = Expr::create(layer_norm_op.get(), {x_var_}, 1);
        layer_norm_expr->setName(expr->name());
        Expr::replace(expr, layer_norm_expr);
        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("FuseLayerNormV4", match, fold);
}

static FuseLayerNormV4 g_fuse_layer_norm_v4;

} // namespace Express
} // namespace MNN
