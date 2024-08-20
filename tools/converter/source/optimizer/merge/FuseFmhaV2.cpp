//
//  FuseFmhaV2.cpp
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

class FuseFmhaV2 {
public:
    FuseFmhaV2();
private:
    EXPRP node_q, node_k, node_v;
    VARP var_q_weight, var_k_weight, var_v_weight;
    VARP fmha_v2_input_;
    int mNumHeads;
};

EXPRP GetFmhaV2BlockCommonNode(EXPRP expr, bool hasReshape = true) {
    auto x = expr;
    EXPRP z;
    if (hasReshape) {
        if (!helpers::IsReshape(expr)) {
            return nullptr;
        }
        z = expr;
        x = z->inputs().at(0)->expr().first;
    }
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

int GetFmhaV2NumHeads(EXPRP expr) {
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

FuseFmhaV2::FuseFmhaV2() {
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
        z = GetFmhaV2BlockCommonNode(v_pre);
        if (z == nullptr) {
            return false;
        }
        mNumHeads = GetFmhaV2NumHeads(z);
        if (mNumHeads == 0) {
            return false;
        }
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
        z = GetFmhaV2BlockCommonNode(k_pre);
        if (z == nullptr) {
            return false;
        }
        
        if (mNumHeads != GetFmhaV2NumHeads(z)) {
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
        z = GetFmhaV2BlockCommonNode(q_pre);
        if (z == nullptr) {
            return false;
        }
        if (mNumHeads != GetFmhaV2NumHeads(z)) {
            return false;
        }
        node_q = z->inputs().at(0)->expr().first;
        if (!helpers::IsMatMul(node_q)) {
            return false;
        }
        
        // QKV -> one source
        if (node_q->inputs().at(0)->expr().first != node_k->inputs().at(0)->expr().first || node_q->inputs().at(0)->expr().first != node_v->inputs().at(0)->expr().first) {
            return false;
        }
        fmha_v2_input_ = node_q->inputs().at(0);
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
            // For target version < 2.8 , don't support fmha_v2
            return false;
        }

        auto* var_q_weight_info    = var_q_weight->getInfo();
        auto* var_k_weight_info    = var_k_weight->getInfo();
        auto* var_v_weight_info    = var_v_weight->getInfo();

        if (!var_q_weight_info || !var_k_weight_info || !var_v_weight_info || var_q_weight_info->size != var_k_weight_info->size || var_q_weight_info->size != var_v_weight_info->size) {
            return false;
        }
        int size = var_q_weight_info->size;
        int C = var_q_weight_info->dim[0];
        int H = mNumHeads;
        int D = var_q_weight_info->dim[1] / H;
        if (var_q_weight_info->dim[1] % H != 0) {
            return false;
        }
        
        // FuseQKV_Weight
        /* [C, H, D]  -> [C, H, 3, D]*/
        auto output = _MatMul(fmha_v2_input_, _Reshape(_Concat({_Reshape(var_q_weight, {C, H, D}), _Reshape(var_k_weight, {C, H, D}), _Reshape(var_v_weight, {C, H, D})}, -1), {C, 3*H*D}), false, false);
        
        std::unique_ptr<MNN::FmhaV2ParamT> param_fmha(new MNN::FmhaV2ParamT);
        param_fmha->heads = mNumHeads;
        
        std::unique_ptr<OpT> fmha_v2_op(new OpT);
        fmha_v2_op->name       = "FmhaV2_" + expr->name();
        fmha_v2_op->type       = OpType_FmhaV2;
        fmha_v2_op->main.type  = OpParameter_FmhaV2Param;
        fmha_v2_op->main.value = param_fmha.release();

        auto fmha_v2_expr = Variable::create(Expr::create(fmha_v2_op.get(), { output }, 1));

        fmha_v2_expr->setName(expr->name());
        Expr::replace(expr, fmha_v2_expr->expr().first);
        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("FuseFmhaV2", match, fold);
}

class FuseSelfAttentionV2 {
public:
    FuseSelfAttentionV2();
private:
    EXPRP node_q, node_k, node_v;
    VARP var_q_weight, var_k_weight, var_v_weight;
    VARP fmha_v2_input_;
    int mNumHeads;
};
    
FuseSelfAttentionV2::FuseSelfAttentionV2() {
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
        
        // whether Einsum/MatMul
        x = z->inputs().at(0)->expr().first;
        if (helpers::IsMatMul(x)) {
            z = x;
        } else {
            return false;
        }
        
        // whether V
        auto qk_pre = z->inputs().at(0)->expr().first;
        auto v_pre = z->inputs().at(1)->expr().first;
        z = GetFmhaV2BlockCommonNode(v_pre);
        if (z == nullptr) {
            return false;
        }
        mNumHeads = GetFmhaV2NumHeads(z);
        if (mNumHeads == 0) {
            return false;
        }
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
        
        //whether add zero
        x = z->inputs().at(0)->expr().first;
        if (helpers::IsBinaryAdd(x)) {
            z = x;
        } else {
            return false;
        }
        
        //add two inputs
        auto x_0 = z->inputs().at(0)->expr().first;
        bool add_0_zero = false;
        if (helpers::IsBinaryMul(x_0)) {
            auto temp_0 = x_0->inputs().at(0)->expr().first;
            auto temp_1 = x_0->inputs().at(1)->expr().first;
            if (helpers::IsConstant(temp_0)) {
                float mul_y = x_0->inputs().at(0)->readMap<float>()[0];
                if(mul_y >= -0.0000001 && mul_y <= 0.0000001) {
                    add_0_zero = true;
                }
            }
            if (helpers::IsConstant(temp_1)) {
                float mul_y = x_0->inputs().at(1)->readMap<float>()[0];
                if(mul_y >= -0.0000001 && mul_y <= 0.0000001) {
                    add_0_zero = true;
                }
            }
        } else {
            return false;
        }
        
        auto x_1 = z->inputs().at(1)->expr().first;
        bool add_1_zero = false;
        if (helpers::IsBinaryMul(x_1)) {
            auto temp_0 = x_1->inputs().at(0)->expr().first;
            auto temp_1 = x_1->inputs().at(1)->expr().first;
            if (helpers::IsConstant(temp_0)) {
                float mul_y = x_1->inputs().at(0)->readMap<float>()[0];
                if(mul_y >= -0.0000001 && mul_y <= 0.0000001) {
                    add_1_zero = true;
                }
            }
            if (helpers::IsConstant(temp_1)) {
                float mul_y = x_1->inputs().at(1)->readMap<float>()[0];
                if(mul_y >= -0.0000001 && mul_y <= 0.0000001) {
                    add_1_zero = true;
                }
            }
        } else {
            return false;
        }
        
        if(add_0_zero && !add_1_zero) {
            x = z->inputs().at(1)->expr().first;
            if(helpers::IsConstant(x->inputs().at(0)->expr().first)) {
                x = x->inputs().at(1)->expr().first;
            } else {
                x = x->inputs().at(0)->expr().first;
            }
        } else if(!add_0_zero && add_1_zero) {
            x = z->inputs().at(0)->expr().first;
            if(helpers::IsConstant(x->inputs().at(0)->expr().first)) {
                x = x->inputs().at(1)->expr().first;
            } else {
                x = x->inputs().at(0)->expr().first;
            }
        } else {
            return false;
        }
        
        //whether matmul
        if (helpers::IsMatMul(x)) {
            z = x;
        } else {
            return false;
        }
        
        auto q_pre = z->inputs().at(0)->expr().first;
        auto k_pre = z->inputs().at(1)->expr().first;
        if(helpers::IsTranspose(k_pre)) {
            k_pre = k_pre->inputs().at(0)->expr().first;
        }
        z = GetFmhaV2BlockCommonNode(k_pre);
        if (z == nullptr) {
            return false;
        }
        
        if (mNumHeads != GetFmhaV2NumHeads(z)) {
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
        z = GetFmhaV2BlockCommonNode(q_pre);
        if (z == nullptr) {
            return false;
        }
        if (mNumHeads != GetFmhaV2NumHeads(z)) {
            return false;
        }
        node_q = z->inputs().at(0)->expr().first;
        if (!helpers::IsMatMul(node_q)) {
            return false;
        }
        
        // QKV -> one source
        if (node_q->inputs().at(0)->expr().first != node_k->inputs().at(0)->expr().first || node_q->inputs().at(0)->expr().first != node_v->inputs().at(0)->expr().first) {
            return false;
        }
        fmha_v2_input_ = node_q->inputs().at(0);
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
            // For target version < 2.8 , don't support fmha_v2
            return false;
        }

        auto* var_q_weight_info    = var_q_weight->getInfo();
        auto* var_k_weight_info    = var_k_weight->getInfo();
        auto* var_v_weight_info    = var_v_weight->getInfo();

        if (!var_q_weight_info || !var_k_weight_info || !var_v_weight_info || var_q_weight_info->size != var_k_weight_info->size || var_q_weight_info->size != var_v_weight_info->size) {
            return false;
        }
        int size = var_q_weight_info->size;
        int C = var_q_weight_info->dim[0];
        int H = mNumHeads;
        int D = var_q_weight_info->dim[1] / H;
        if (var_q_weight_info->dim[1] % H != 0) {
            return false;
        }
        
        // FuseQKV_Weight
        /* [C, H, D]  -> [C, H, 3, D]*/
        auto output = _MatMul(fmha_v2_input_, _Reshape(_Concat({_Reshape(var_q_weight, {C, H, D}), _Reshape(var_k_weight, {C, H, D}), _Reshape(var_v_weight, {C, H, D})}, -1), {C, H*3*D}), false, false);
        
        std::unique_ptr<MNN::FmhaV2ParamT> param_fmha(new MNN::FmhaV2ParamT);
        param_fmha->heads = mNumHeads;
        
        std::unique_ptr<OpT> fmha_v2_op(new OpT);
        fmha_v2_op->name       = "FmhaV2_" + expr->name();
        fmha_v2_op->type       = OpType_FmhaV2;
        fmha_v2_op->main.type  = OpParameter_FmhaV2Param;
        fmha_v2_op->main.value = param_fmha.release();

        auto fmha_v2_expr = Variable::create(Expr::create(fmha_v2_op.get(), { output }, 1));

        fmha_v2_expr->setName(expr->name());
        Expr::replace(expr, fmha_v2_expr->expr().first);
        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("FuseSelfAttentionV2", match, fold);
}


class FuseSelfFmhaV2 {
public:
    FuseSelfFmhaV2();
private:
    VARP self_fmha_v2_input_;
    int mNumHeads;
};

FuseSelfFmhaV2::FuseSelfFmhaV2() {
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
        // whether Einsum/MatMul
        x = z->inputs().at(0)->expr().first;
        if (helpers::IsMatMul(x)) {
            z = x;
        } else {
            return false;
        }
        
        // whether V
        auto qk_pre = z->inputs().at(0)->expr().first;
        auto v_pre = z->inputs().at(1)->expr().first;
        z = GetFmhaV2BlockCommonNode(v_pre, false);
        if (z == nullptr) {
            return false;
        }
        
        mNumHeads = GetFmhaV2NumHeads(z);
        if (mNumHeads == 0) {
            return false;
        }
        // whether split
        auto common_split = z->inputs().at(0)->expr().first;
        if (!helpers::IsSlice(common_split)) {
            return false;
        }

        z = qk_pre;
        // whether softmax
        if (!helpers::IsSoftmax(z)) {
            return false;
        }
        
        //whether add
        x = z->inputs().at(0)->expr().first;
        if (helpers::IsBinaryAdd(x)) {
            z = x;
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
        z = GetFmhaV2BlockCommonNode(k_pre, false);
        if (z == nullptr) {
            return false;
        }
        if (mNumHeads != GetFmhaV2NumHeads(z)) {
            return false;
        }
        
        if (common_split != z->inputs().at(0)->expr().first) {
            return false;
        }
        // whether mul(scale)
        if (helpers::IsBinaryMul(q_pre)) {
            q_pre = q_pre->inputs().at(0)->expr().first;
        }
        
        z = GetFmhaV2BlockCommonNode(q_pre, false);
        if (z == nullptr) {
            return false;
        }
        if (mNumHeads != GetFmhaV2NumHeads(z)) {
            return false;
        }
        
        if (common_split != z->inputs().at(0)->expr().first) {
            return false;
        }
        self_fmha_v2_input_ = common_split->inputs().at(0);
        
        return true;
    };

    auto fold = [this](EXPRP expr) -> bool {
        auto config = Global<modelConfig>::Get();
        auto version = config->targetVersion;
        if (version < 2.8f) {
            // For target version < 2.8 , don't support fmha_v2
            return false;
        }

        std::unique_ptr<MNN::FmhaV2ParamT> param_fmha(new MNN::FmhaV2ParamT);
        param_fmha->heads = mNumHeads;
        
        std::unique_ptr<OpT> fmha_v2_op(new OpT);
        fmha_v2_op->name       = "Self_FmhaV2_" + expr->name();
        fmha_v2_op->type       = OpType_FmhaV2;
        fmha_v2_op->main.type  = OpParameter_FmhaV2Param;
        fmha_v2_op->main.value = param_fmha.release();

        auto fmha_v2_expr = Variable::create(Expr::create(fmha_v2_op.get(), { self_fmha_v2_input_ }, 1));

        fmha_v2_expr->setName(expr->name());
        Expr::replace(expr, fmha_v2_expr->expr().first);
        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("FuseSelfFmhaV2", match, fold);
}


static FuseFmhaV2 g_fuse_fmhaV2;
static FuseSelfFmhaV2 g_fuse_self_fmhaV2;
static FuseSelfAttentionV2 g_fuse_self_attentionV2;

} // namespace Express
} // namespace MNN
