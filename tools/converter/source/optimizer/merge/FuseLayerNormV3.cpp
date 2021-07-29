//
//  FuseLayerNormV3.cpp
//  MNNConverter
//
//  Created by MNN on 2021/06/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <unordered_map>

#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"
#include "MergeHelpers.hpp"

namespace MNN {
namespace Express {

class FuseLayerNormV3 {
public:
    FuseLayerNormV3();

private:
    VARP x_var_;
    std::vector<int> mAxis;
    bool has_gamma_ = false;
    VARP gamma_var_;
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

FuseLayerNormV3::FuseLayerNormV3() {
    auto match = [this](EXPRP expr) -> bool {
        if (!expr->get() || !helpers::IsBinaryAdd(expr)) {
            return false;
        }
        EXPRP mul_1 = expr->inputs().at(0)->expr().first;
        EXPRP mul_2 = expr->inputs().at(1)->expr().first;
        if (!helpers::IsBinaryMul(mul_1) || !helpers::IsBinaryMul(mul_2)) {
            return false;
        }
        EXPRP x   = mul_1->inputs().at(0)->expr().first;
        EXPRP rsqrt = mul_1->inputs().at(1)->expr().first;
        if (helpers::IsBinaryMul(rsqrt)) {
            gamma_var_ = rsqrt->inputs().at(1);
            if (!helpers::IsConstant(gamma_var_->expr().first)) {
                return false;
            }
            rsqrt = rsqrt->inputs().at(0)->expr().first;
            has_gamma_ = true;
        } else if (helpers::IsUnaryRsqrt(rsqrt)) {
            has_gamma_ = false;
        } else {
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
        EXPRP neg = mul_2->inputs().at(0)->expr().first;
        if (!helpers::IsUnaryNeg(neg) || mul_2->inputs().at(1).get() != mul_1->inputs().at(1).get()) {
            return false;
        }
        if (neg->inputs().at(0).get() != square_diff->inputs().at(1).get()) {
            return false;
        }
        // Cache the variables to build layer normalization.
        x_var_       = x_var;
        mAxis        = variance_axis;
        epsilon_var_ = add->inputs().at(1);
        return true;
    };

    auto fold = [this](EXPRP expr) -> bool {
        auto config = Global<modelConfig>::Get();
        auto version = config->targetVersion;
        std::unique_ptr<MNN::LayerNormT> layer_norm(new MNN::LayerNormT);
        layer_norm->axis = mAxis;
        layer_norm->epsilon = epsilon_var_->readMap<float>()[0];
        if (has_gamma_) {
            auto* gamma_info   = gamma_var_->getInfo();
            const float* gamma = gamma_var_->readMap<float>();
            int size = gamma_info->size;
            layer_norm->gamma.resize(size);
            layer_norm->beta.resize(size);
            memcpy(layer_norm->gamma.data(), gamma, size * sizeof(float));
            memset(layer_norm->beta.data(), 0, size);
        } else if (version < 1.3f) {
            // For target version < 1.3 , don't support layernorm without gamma beta
            return false;
        }
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
    TemplateMerge::getInstance("Merge").insertTemplate("FuseLayerNormV3", match, fold);
}

static FuseLayerNormV3 g_fuse_layer_norm_v3;

} // namespace Express
} // namespace MNN
