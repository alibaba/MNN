//
//  FuseLayerNormV2.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <unordered_map>

#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"
#include "MergeHelpers.hpp"

namespace MNN {
namespace Express {

class FuseLayerNormV2 {
public:
    FuseLayerNormV2();

private:
    VARP x_var_;
    std::vector<int> mAxis;
    VARP gamma_var_;
    VARP beta_var_;
    VARP epsilon_var_;
};
static std::vector<int> _getReduceDims(EXPRP variance, bool& success) {
    std::vector<int> varianceDims;
    if (variance->inputs().size() >= 2) {
        auto variance_axis = variance->inputs().at(1);
        auto variance_axis_info = variance_axis->getInfo();
        auto variance_axis_ptr = variance_axis->readMap<int>();
        if (nullptr == variance_axis_info || nullptr == variance_axis_ptr) {
            success = false;
            return varianceDims;
        }
        varianceDims.resize(variance_axis_info->size);
        ::memcpy(varianceDims.data(), variance_axis_ptr, variance_axis_info->size*sizeof(int));
    } else {
        auto red = variance->get()->main_as_ReductionParam();
        if (red->dim() != nullptr) {
            varianceDims.resize(red->dim()->size());
            ::memcpy(varianceDims.data(), red->dim()->data(), varianceDims.size() * sizeof(int));
        }
    }
    success = true;
    return varianceDims;
}

FuseLayerNormV2::FuseLayerNormV2() {
    auto match = [this](EXPRP expr) -> bool {
        if (!expr->get() || !helpers::IsBinaryAdd(expr)) {
            return false;
        }
        EXPRP mul_1 = expr->inputs().at(0)->expr().first;
        EXPRP sub   = expr->inputs().at(1)->expr().first;
        if (!helpers::IsBinaryMul(mul_1) || !helpers::IsBinarySub(sub)) {
            return false;
        }
        EXPRP x   = mul_1->inputs().at(0)->expr().first;
        EXPRP mul = mul_1->inputs().at(1)->expr().first;
        if (!helpers::IsBinaryMul(mul)) {
            return false;
        }
        EXPRP rsqrt = mul->inputs().at(0)->expr().first;
        EXPRP gamma = mul->inputs().at(1)->expr().first;
        if (!helpers::IsUnaryRsqrt(rsqrt) || !helpers::IsConstant(gamma)) {
            return false;
        }
        EXPRP add = rsqrt->inputs().at(0)->expr().first;
        if (!helpers::IsBinaryAdd(add)) {
            return false;
        }
        EXPRP variance = add->inputs().at(0)->expr().first;
        EXPRP epsilon  = add->inputs().at(1)->expr().first;
        if (!helpers::IsReductionMean(variance) || !helpers::IsConstant(epsilon)) {
            return false;
        }
        bool success = true;
        std::vector<int> variance_axis = _getReduceDims(variance, success);
        if (!success) {
            return false;
        }
        EXPRP square_diff   = variance->inputs().at(0)->expr().first;
        if (!helpers::IsBinarySquaredDifference(square_diff)) {
            return false;
        }

        VARP x_var = square_diff->inputs().at(0);
        if (x_var.get() != mul_1->inputs().at(0).get()) {
            return false;
        }
        EXPRP mean = square_diff->inputs().at(1)->expr().first;
        if (!helpers::IsReductionMean(mean)) {
            return false;
        }
        if (x_var.get() != mean->inputs().at(0).get()) {
            return false;
        }
        std::vector<int> mean_axis = _getReduceDims(mean, success);
        if (!success) {
            return false;
        }
        if (mean_axis.size() != variance_axis.size()) {
            return false;
        }
        for (int i=0; i<mean_axis.size(); ++i) {
            if (mean_axis[i] != variance_axis[i]) {
                return false;
            }
        }
        EXPRP beta  = sub->inputs().at(0)->expr().first;
        EXPRP mul_2 = sub->inputs().at(1)->expr().first;
        if (!helpers::IsConstant(beta) || !helpers::IsBinaryMul(mul_2)) {
            return false;
        }
        if (mul_2->inputs().at(0).get() != square_diff->inputs().at(1).get()) {
            return false;
        }
        if (mul_2->inputs().at(1).get() != mul_1->inputs().at(1).get()) {
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
        mAxis        = variance_axis;
        gamma_var_   = mul->inputs().at(1);
        beta_var_    = sub->inputs().at(0);
        epsilon_var_ = add->inputs().at(1);
        return true;
    };

    auto fold = [this](EXPRP expr) -> bool {
        std::unique_ptr<MNN::LayerNormT> layer_norm(new MNN::LayerNormT);
        layer_norm->axis = mAxis;
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
    TemplateMerge::getInstance("Merge").insertTemplate("FuseLayerNormV2", match, fold);
}

static FuseLayerNormV2 g_fuse_layer_norm_v2;

} // namespace Express
} // namespace MNN
