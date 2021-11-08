//
//  FuseLayerNorm.cpp
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

class FuseLayerNorm {
public:
    FuseLayerNorm();

private:
    std::vector<int> axis_var_;
    VARP x_var_;
    VARP gamma_var_;
    VARP beta_var_;
    VARP epsilon_var_;
};

FuseLayerNorm::FuseLayerNorm() {
    auto match = [this](EXPRP expr) -> bool {
        if (!expr->get() || !helpers::IsBinaryAdd(expr)) {
            return false;
        }
        EXPRP mul_3 = expr->inputs().at(0)->expr().first;
        EXPRP beta  = expr->inputs().at(1)->expr().first;
        if (!helpers::IsBinaryMul(mul_3) || !helpers::IsConstant(beta)) {
            return false;
        }
        EXPRP mul_2 = mul_3->inputs().at(0)->expr().first;
        EXPRP gamma = mul_3->inputs().at(1)->expr().first;
        if (!helpers::IsBinaryMul(mul_2) || !helpers::IsConstant(gamma)) {
            return false;
        }
        EXPRP sub_2 = mul_2->inputs().at(0)->expr().first;
        EXPRP rsqrt = mul_2->inputs().at(1)->expr().first;
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
        VARP sub_2_var = mul_2->inputs().at(0);
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

        // TODO(): Check if axis is satisfied or not.
        // auto* x_info = x_var->getInfo();
        // if (!x_info) {
        //     return false;
        // }
        // const int rank = x_info->dim.size();
        // auto* axis_info = axis_var->getInfo();
        // if (!axis_info) {
        //     return false;
        // }
        // std::vector<int> axes(axis_info->size);
        // for (int i = 0; i < axis_info->size; ++i) {
        //     axes[i] = axis_var->readMap<int>()[i];
        //     if (axes[i] < 0) {
        //         axes[i] += rank;
        //     }
        // }
        // std::sort(axes.begin(), axes.end());
        // for (int i = 0; i < axes.size(); ++i) {
        //     if (axes.at(i) != rank - axes.size() + i) {
        //         return false;
        //     }
        // }

        // Cache the variables to build layer normalization.
        x_var_       = x_var;
        gamma_var_   = mul_3->inputs().at(1);
        beta_var_    = expr->inputs().at(1);
        epsilon_var_ = add_2->inputs().at(1);
        return true;
    };

    auto fold = [this](EXPRP expr) -> bool {
        std::unique_ptr<MNN::LayerNormT> layer_norm(new MNN::LayerNormT);
        layer_norm->axis = axis_var_;
        layer_norm->epsilon = epsilon_var_->readMap<float>()[0];

        auto* gamma_info   = gamma_var_->getInfo();
        auto* beta_info    = beta_var_->getInfo();
        const float* gamma = gamma_var_->readMap<float>();
        const float* beta  = beta_var_->readMap<float>();
        if (!gamma_info || !beta_info || !gamma || !beta || gamma_info->size != beta_info->size) {
            return false;
        }
        int size = gamma_info->size;
        layer_norm->gamma.resize(size);
        layer_norm->beta.resize(size);
        memcpy(layer_norm->gamma.data(), gamma, size * sizeof(float));
        memcpy(layer_norm->beta.data(), beta, size * sizeof(float));

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
    TemplateMerge::getInstance("Merge").insertTemplate("FuseLayerNorm", match, fold);
}

static FuseLayerNorm g_fuse_layer_norm;

} // namespace Express
} // namespace MNN
