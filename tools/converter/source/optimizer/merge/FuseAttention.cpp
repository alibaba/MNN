//
//  FuseAttention.cpp
//  MNNConverter
//
//  Created by MNN on 2024/03/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <unordered_map>

#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"
#include "MergeHelpers.hpp"

namespace MNN {
namespace Express {

class FuseAttention {
public:
    FuseAttention();
private:
    VARP query, key, value, mask;
};

static EXPRP is_gqa(EXPRP& x) {
    if (!helpers::IsReshape(x)) {
        return x;
    }
    auto y = x->inputs().at(0)->expr().first;
    if (!helpers::IsBroadcastTo(y)) {
        return x;
    }
    y = y->inputs().at(0)->expr().first;
    if (!helpers::IsUnsqueeze(y)) {
        return x;
    }
    y = y->inputs().at(0)->expr().first;
    return y;
}

FuseAttention::FuseAttention() {
    auto match = [this](EXPRP expr) -> bool {
        auto config = Global<modelConfig>::Get();
        if(!config->transformerFuse) {
            return false;
        }
        // transpose
        if (!helpers::IsReshape(expr)) {
            return false;
        }

        expr = expr->inputs().at(0)->expr().first;
        if (!helpers::IsTranspose(expr)) {
            return false;
        }

        EXPRP x, y;
        // softmax @ v
        auto matmul = expr->inputs().at(0)->expr().first;
        if (!helpers::IsMatMul(matmul)) {
            return false;
        }
        y = matmul->inputs().at(1)->expr().first;
        y = is_gqa(y);
        // transpose
        if (!helpers::IsTranspose(y)) {
            return false;
        }
        // concat
        y = y->inputs().at(0)->expr().first;
        if (!helpers::IsConcat(y)) {
            return false;
        }
        // value
        value = y->inputs().at(1);

        // softmax
        x = matmul->inputs().at(0)->expr().first;
        if (helpers::IsCast(x)) {
            x = x->inputs().at(0)->expr().first;
        }
        if (!helpers::IsSoftmax(x)) {
            return false;
        }
        // mask
        x = x->inputs().at(0)->expr().first;
        if (helpers::IsSelect(x)) {
            mask = x->inputs().at(0);
            x = x->inputs().at(1)->expr().first;
        } else if (helpers::IsBinaryAdd(x)) {
            mask = x->inputs().at(1);
            x = x->inputs().at(0)->expr().first;
        } else {
            return false;
        }

        // div
        if (helpers::IsCast(x)) {
            x = x->inputs().at(0)->expr().first;
        }
        if (!helpers::IsBinaryOp(x)) {
            return false;
        }
        // q @ k
        x = x->inputs().at(0)->expr().first;
        if (!helpers::IsMatMul(x)) {
            return false;
        }
        // transpose
        auto z = x->inputs().at(0)->expr().first;
        if (!helpers::IsTranspose(z)) {
            return false;
        }
        // query
        query = z->inputs().at(0);

        y = x->inputs().at(1)->expr().first;
        // transpose
        y = is_gqa(y);
        if (!helpers::IsTranspose(y)) {
            return false;
        }
        // concat
        y = y->inputs().at(0)->expr().first;
        if (!helpers::IsConcat(y)) {
            return false;
        }
        // key
        key = y->inputs().at(1);
        return true;
    };

    auto fold = [this](EXPRP expr) -> bool {
        auto config = Global<modelConfig>::Get();
        auto version = config->targetVersion;
        if (version < 2.8f) {
            // For target version < 2.8 , don't support fmha_v2
            return false;
        }
        if (expr->name().size() > 0) {
            MNN_PRINT("Fuse Attention as %s\n", expr->name().c_str());
        }

        std::unique_ptr<OpT> attention(new OpT);
        attention->name       = "Attention" + expr->name();
        attention->type       = OpType_Attention;
        attention->main.type  = OpParameter_AttentionParam;
        attention->main.value = new AttentionParamT;

        auto attention_expr = Variable::create(Expr::create(attention.get(), {query, key, value, mask}, 1));

        attention_expr->setName(expr->name());
        Expr::replace(expr, attention_expr->expr().first);
        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("FuseAttention", match, fold);
}

class RemovePastKeyValue {
public:
    RemovePastKeyValue();
private:
    VARP kv_in;
};

RemovePastKeyValue::RemovePastKeyValue() {
    auto match = [this](EXPRP expr) -> bool {
        auto config = Global<modelConfig>::Get();
        if(!config->transformerFuse) {
            return false;
        }
        /*
           llm: out <- stack [concat + unsqueeze] <- stack [concat + unsqueeze] <- concat <- gatherv2 <- gatherv2 <- in
         block: out <------------------------------- stack [concat + unsqueeze] <- concat <- gatherv2 <------------- in
         */
        if (!helpers::IsConcat(expr)) {
            return false;
        }
        expr = expr->inputs().at(0)->expr().first;
        if (!helpers::IsUnsqueeze(expr)) {
            return false;
        }
        expr = expr->inputs().at(0)->expr().first;
        if (!helpers::IsConcat(expr) && expr->inputs().size() == 2) {
            return false;
        }
        expr = expr->inputs().at(0)->expr().first;
        // llm model
        if (helpers::IsUnsqueeze(expr)) {
            // concat [past_k, k]
            expr = expr->inputs().at(0)->expr().first;
            if (!helpers::IsConcat(expr) && expr->inputs().size() == 2) {
                return false;
            }
            // gatherv2
            expr = expr->inputs().at(0)->expr().first;
            if (!helpers::IsGatherV2(expr)) {
                return false;
            }
            // gatherv2
            expr = expr->inputs().at(0)->expr().first;
            if (!helpers::IsGatherV2(expr)) {
                return false;
            }
            kv_in = expr->inputs().at(0);
            if (!kv_in->expr().first->inputs().empty()) {
                return false;
            }
            return true;
        }
        // block model
        if (helpers::IsGatherV2(expr)) {
            kv_in = expr->inputs().at(0);
            if (!kv_in->expr().first->inputs().empty()) {
                return false;
            }
            return true;
        }
        return false;
    };

    auto fold = [this](EXPRP expr) -> bool {
        auto config = Global<modelConfig>::Get();
        auto version = config->targetVersion;
        if (version < 2.8f) {
            // For target version < 2.8 , don't support fmha_v2
            return false;
        }
        if (!expr->name().empty()) {
            MNN_PRINT("Remove past KV for %s\n", expr->name().c_str());
        }

        // past-kv remove
        std::unique_ptr<OpT> reshape(new OpT);
        reshape->name       = expr->name();
        reshape->type       = OpType_Reshape;
        reshape->main.type  = OpParameter_Reshape;
        auto reshape_t = new ReshapeT;
        reshape_t->dims = {-1};
        reshape->main.value = reshape_t;
        auto copy_expr = Variable::create(Expr::create(reshape.get(), {kv_in}, 1));
        Expr::replace(expr, copy_expr->expr().first);
        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("RemovePastKeyValue", match, fold);
}


static FuseAttention g_fuse_attenion;
static RemovePastKeyValue g_remove_kv;

} // namespace Express
} // namespace MNN
