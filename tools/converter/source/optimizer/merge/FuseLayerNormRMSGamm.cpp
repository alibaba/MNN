//
//  FuseLayerNorm.cpp
//  MNNConverter
//
//  Created by MNN on 2024/01/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <unordered_map>

#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"
#include "MergeHelpers.hpp"

namespace MNN {
namespace Express {
class FuseLayerNormRMSGamma {
public:
    FuseLayerNormRMSGamma();

private:
    std::vector<int> axis_var_;
    VARP x_var_;
    VARP epsilon_var_;
};
FuseLayerNormRMSGamma::FuseLayerNormRMSGamma() {
    auto match = [this](EXPRP expr) -> bool {
        if (!expr->get() || !helpers::IsBinaryMul(expr)) {
            return false;
        }
        auto mulexpr = expr->inputs().at(1)->expr().first;
        auto conexpr = expr->inputs().at(0)->expr().first;
        if (helpers::IsLayerNorm(mulexpr) && helpers::IsConstant(conexpr)) {
            std::unique_ptr<OpT> op(mulexpr->get()->UnPack());
            auto params = op->main.AsLayerNorm();
            if (!params->gamma.empty() || !params->beta.empty()) {
                return false;
            }
            return true;
        }
        if (helpers::IsLayerNorm(conexpr) && helpers::IsConstant(mulexpr)) {
            std::unique_ptr<OpT> op(conexpr->get()->UnPack());
            auto params = op->main.AsLayerNorm();
            if (!params->gamma.empty() || !params->beta.empty()) {
                return false;
            }
            return true;
        }
        return false;
    };

    auto fold = [this](EXPRP expr) -> bool {
        std::unique_ptr<MNN::LayerNormT> layer_norm(new MNN::LayerNormT);
        auto mulexpr = expr->inputs().at(1)->expr().first;
        auto conexpr = expr->inputs().at(0)->expr().first;
        int k = 0;
        if (helpers::IsConstant(mulexpr) && helpers::IsLayerNorm(conexpr)) {
            k = 1;
        }
        
            
        mulexpr = expr->inputs().at(1-k)->expr().first;
        std::unique_ptr<OpT> op(mulexpr->get()->UnPack());
        auto params  = op->main.AsLayerNorm();
        std::unique_ptr<MNN::LayerNormT> layernorm(new MNN::LayerNormT);
        layernorm->axis = params->axis;
        layernorm->epsilon = params->epsilon;
        layernorm->useRMSNorm = params->useRMSNorm;
        auto gammaVar = expr->inputs().at(k);
        auto gammasize = gammaVar->getInfo()->size;
        // if (expr->inputs().at(1 - k)->getInfo() == nullptr) {
        //     return false;
        // }
        // auto reducesize = expr->inputs().at(1 - k)->getInfo()->dim[expr->inputs().at(1 - k)->getInfo()->dim.size() - 1];
        // if (reducesize != gammasize) {
        //     return false;
        // }
        layernorm->gamma.resize(gammasize);
        ::memcpy(layernorm->gamma.data(), gammaVar->readMap<float>(), gammasize * sizeof(float));
        layernorm->beta.resize(gammasize);
        std::unique_ptr<OpT> newOp(new OpT);
        newOp->name       = mulexpr->name();
        newOp->type       = OpType_LayerNorm;
        newOp->main.type  = OpParameter_LayerNorm;
        newOp->main.value = layernorm.release();

        EXPRP layer_norm_expr = Expr::create(newOp.get(), mulexpr->inputs(), 1);
        layer_norm_expr->setName(expr->name());
        Expr::replace(expr, layer_norm_expr);
        return true;
        
    };
    TemplateMerge::getInstance("Merge").insertTemplate("FuseLayerNormRMSGamma", match, fold);
}
static FuseLayerNormRMSGamma g_fuse_layer_norm_rms_gamma;
} // namespace Express
} // namespace MNN
