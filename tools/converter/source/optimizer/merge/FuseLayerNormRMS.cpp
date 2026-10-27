//
//  FuseLayerNorm.cpp
//  MNNConverter
//
//  Created by MNN on 2024/01/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cmath>
#include <unordered_map>

#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"
#include "MergeHelpers.hpp"

namespace MNN {
namespace Express {

// True when var is a single-element float constant equal to expect.
static bool isConstantScalar(VARP var, float expect) {
    if (nullptr == var || !helpers::IsConstant(var->expr().first)) {
        return false;
    }
    auto info = var->getInfo();
    if (nullptr == info || 1 != info->size || halide_type_float != info->type.code || 32 != info->type.bits) {
        return false;
    }
    auto value = var->readMap<float>();
    return nullptr != value && fabsf(value[0] - expect) < 1e-6f;
}

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

class FuseLayerNormRMS {
public:
    FuseLayerNormRMS();

private:
    std::vector<int> axis_var_;
    VARP x_var_;
    VARP epsilon_var_;
};

FuseLayerNormRMS::FuseLayerNormRMS() {
    auto match = [this](EXPRP expr) -> bool {
        if (!expr->get() || !helpers::IsBinaryMul(expr)) {
            return false;
        }
        EXPRP rsqrt = expr->inputs().at(1)->expr().first;
        if (helpers::IsBinaryRealDiv(rsqrt)) {
            rsqrt = rsqrt->inputs().at(1)->expr().first;
            if (!helpers::IsUnarySqrt(rsqrt)) {
                return false;
            }
        } else if (helpers::IsBinaryPow(rsqrt)) {
            // pow(y, -0.5) is rsqrt(y) for the strictly positive y this pattern feeds it.
            // HuggingFace's gemma4 RMSNorm emits this spelling instead of torch.rsqrt.
            if (!isConstantScalar(rsqrt->inputs().at(1), -0.5f)) {
                return false;
            }
        } else if (!helpers::IsUnaryRsqrt(rsqrt)) {
            return false;
        }
        EXPRP add = rsqrt->inputs().at(0)->expr().first;
        if (!helpers::IsBinaryAdd(add)) {
            return false;
        }
        EXPRP mean = add->inputs().at(0)->expr().first;
        EXPRP epsilon = add->inputs().at(1)->expr().first;
        if (!helpers::IsReductionMean(mean) || !helpers::IsConstant(epsilon)) {
            return false;
        }
        auto axisLoad = loadAxisFromReduction(mean, axis_var_);
        if (!axisLoad) {
            return false;
        }
        EXPRP pow = mean->inputs().at(0)->expr().first;
        if (!helpers::IsUnarySquare(pow)) {
            return false;
        }
        VARP x_var    = pow->inputs().at(0);
        if (expr->inputs().at(0).get() != x_var.get()) {
            return false;
        }

        // Cache the variables to build layer normalization.
        x_var_       = x_var;
        epsilon_var_ = add->inputs().at(1);
        return true;
    };

    auto fold = [this](EXPRP expr) -> bool {
        std::unique_ptr<MNN::LayerNormT> layer_norm(new MNN::LayerNormT);
        layer_norm->axis = axis_var_;
        layer_norm->epsilon = epsilon_var_->readMap<float>()[0];
        layer_norm->useRMSNorm = true;

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
    TemplateMerge::getInstance("Merge").insertTemplate("FuseLayerNormRMS", match, fold);
}

static FuseLayerNormRMS g_fuse_layer_norm_rms;

} // namespace Express
} // namespace MNN
