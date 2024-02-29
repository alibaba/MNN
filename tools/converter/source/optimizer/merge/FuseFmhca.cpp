//
//  FuseFmhca.cpp
//  MNNConverter
//
//  Created by MNN on 2024/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <unordered_map>

#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"
#include "MergeHelpers.hpp"

namespace MNN {
namespace Express {

class FuseFmhca {
public:
    FuseFmhca();
private:
    EXPRP node_q, node_k, node_v;
    VARP var_q_weight, var_k_weight, var_v_weight;
    VARP fmhca_kv_input_, fmhca_q_input_;
    int mNumHeads;
};

EXPRP GetFmhcaBlockAboveNode(EXPRP expr) {
    if (!helpers::IsReshape(expr)) {
        return nullptr;
    }
    auto z = expr;
    auto x = z->inputs().at(0)->expr().first;
    if (!helpers::IsTranspose(x)) {
        return nullptr;
    }
    z = x;
    x = z->inputs().at(0)->expr().first;
    if (!helpers::IsReshape(x)) {
        return nullptr;
    }
    z = x;
    return z;
}

int GetFmhcaNumHeads(EXPRP expr) {
    if (!helpers::IsReshape(expr)) {
        return 0;
    }
    auto z = expr;
    auto x = z->inputs().at(1)->expr().first;
    if (!helpers::IsConcat(x)) {
        return 0;
    }
    z = x;
    x = z->inputs().at(2)->expr().first;
    if (!helpers::IsConstant(x)) {
        return 0;
    }
    auto var_num_head = z->inputs().at(2);
    return var_num_head->readMap<int32_t>()[0];
}

FuseFmhca::FuseFmhca() {
    auto match = [this](EXPRP expr) -> bool {
        auto config = Global<modelConfig>::Get();
        if(!config->transformerFuse) {
            return false;
        }
        // whether reshape
        if (!expr->get() || !helpers::IsReshape(expr)) {
            return false;
        }
        
        EXPRP x, y, z;
        
        // whether transpose
        x = expr->inputs().at(0)->expr().first;
        if (!expr->get() || !helpers::IsTranspose(x)) {
            return false;
        }
        z = x;
        
        // whether reshape
        x = z->inputs().at(0)->expr().first;
        if (!helpers::IsReshape(x)) {
            return false;
        }
        z = x;
        
        // whether cast
        x = z->inputs().at(0)->expr().first;
        if (helpers::IsCast(x)) {
            z = x->inputs().at(0)->expr().first;
        } else {
            z = x;
        }
        
        // whether scatternd
        while (z->inputs().size() >= 3 && helpers::IsScatterNd(z)) {
            z = z->inputs().at(1)->expr().first;
        }
        
        // whether Einsum/MatMul
        x = z->inputs().at(0)->expr().first;
        if (!x->get()) {
            return false;
        }
        x = x->inputs().at(0)->expr().first;

        if (helpers::IsMatMul(x)) {
            z = x;
        } else {
            return false;
        }
        
        // whether V
        auto qk_pre = z->inputs().at(0)->expr().first;
        auto v_pre = z->inputs().at(1)->expr().first;
        z = GetFmhcaBlockAboveNode(v_pre);
        if (z == nullptr) {
            return false;
        }
        mNumHeads = GetFmhcaNumHeads(z);

        node_v = z->inputs().at(0)->expr().first;
        if (!helpers::IsMatMul(node_v)) {
            return false;
        }

        // whether cast
        if (helpers::IsCast(qk_pre)) {
            qk_pre = qk_pre->inputs().at(0)->expr().first;
        }
        z = qk_pre;
        // whether softmax
        if (!helpers::IsSoftmax(z)) {
            return false;
        }
        //whether matmul
        x = z->inputs().at(0)->expr().first;
        if (helpers::IsMatMul(x)) {
            z = x;
        } else {
            return false;
        }
        
        auto q_pre = z->inputs().at(0)->expr().first;
        auto k_pre = z->inputs().at(1)->expr().first;
        z = GetFmhcaBlockAboveNode(k_pre);
        if (z == nullptr) {
            return false;
        }
        if (mNumHeads != GetFmhcaNumHeads(z)) {
            return false;
        }
        node_k = z->inputs().at(0)->expr().first;
        // whether mul(scale)
        if (helpers::IsBinaryMul(node_k)) {
            node_k = node_k->inputs().at(0)->expr().first;
        }
        
        if (!helpers::IsMatMul(node_k)) {
            return false;
        }
        
        // whether slice
        if (helpers::IsSlice(q_pre)) {
            q_pre = q_pre->inputs().at(0)->expr().first;
        }
        z = GetFmhcaBlockAboveNode(q_pre);
        if (z == nullptr) {
            return false;
        }
        if (mNumHeads != GetFmhcaNumHeads(z)) {
            return false;
        }
        
        fmhca_q_input_ = z->inputs().at(0);
        node_q = z->inputs().at(0)->expr().first;
        if (!helpers::IsMatMul(node_q)) {
            return false;
        }
        
        // KV -> one source, Q -> different
        if (node_k->inputs().at(0)->expr().first != node_v->inputs().at(0)->expr().first || node_q->inputs().at(0)->expr().first == node_k->inputs().at(0)->expr().first) {
            return false;
        }
        fmhca_kv_input_ = node_k->inputs().at(0);
        var_q_weight = node_q->inputs().at(1);
        var_k_weight = node_k->inputs().at(1);
        var_v_weight = node_v->inputs().at(1);
        
        if(!helpers::IsConstant(var_q_weight->expr().first) || !helpers::IsConstant(var_k_weight->expr().first) || !helpers::IsConstant(var_v_weight->expr().first)) {
            return false;
        }
        return true;
    };

    auto fold = [this](EXPRP expr) -> bool {
        auto config = Global<modelConfig>::Get();
        auto version = config->targetVersion;
        if (version < 2.8f) {
            // For target version < 2.8 , don't support fmhca
            return false;
        }
        auto* var_k_weight_info    = var_k_weight->getInfo();
        auto* var_v_weight_info    = var_v_weight->getInfo();

        if (!var_k_weight_info || !var_v_weight_info || var_k_weight_info->size != var_v_weight_info->size) {
            return false;
        }
        int size = var_k_weight_info->size;
        int C = var_k_weight_info->dim[0];
        int H = mNumHeads;
        int D = var_k_weight_info->dim[1] / H;
        if (var_k_weight_info->dim[1] % H != 0) {
            return false;
        }
        
        auto output = _MatMul(fmhca_kv_input_, _Reshape(_Concat({_Reshape(var_k_weight, {C, H, D}), _Reshape(var_v_weight, {C, H, D})}, -1), {C, 2*H*D}), false, false);
        
        
        std::unique_ptr<MNN::FmhcaParamT> param_fmhca(new MNN::FmhcaParamT);
        param_fmhca->heads = mNumHeads;
        
        std::unique_ptr<OpT> fmhca_op(new OpT);
        fmhca_op->name       = "Fmhca_" + expr->name();
        fmhca_op->type       = OpType_Fmhca;
        fmhca_op->main.type  = OpParameter_FmhcaParam;
        fmhca_op->main.value = param_fmhca.release();
        
        auto fmhca_expr = Variable::create(Expr::create(fmhca_op.get(), {fmhca_q_input_, output}, 1));
        
        fmhca_expr->setName(expr->name());

        Expr::replace(expr, fmhca_expr->expr().first);
        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("FuseFmhca", match, fold);
}

static FuseFmhca g_fuse_Fmhca;

} // namespace Express
} // namespace MNN
