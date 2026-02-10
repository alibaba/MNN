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
    VARP var_q, var_k, var_v;
    VARP var_q_weight, var_k_weight, var_v_weight;
    int mNumHeads;
};

EXPRP GetFmhaV2BlockCommonNode(EXPRP expr, bool hasReshape = true) {
    auto x = expr;
    EXPRP z, res;
    // 3 dimension or 4 dimension both ok
    if (helpers::IsReshape(expr)) {
        z = expr;
        x = z->inputs().at(0)->expr().first;
    }

    if (helpers::IsTranspose(x)) {
        z = x;
        x = z->inputs().at(0)->expr().first;
        if (helpers::IsReshape(x)) {
            res = x;
            x = x->inputs().at(0)->expr().first;
        }
    }
    if (!helpers::IsTranspose(x)) {
        return res;
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
    int head_num_idx = z->inputs().size() - 2;
    MNN_ASSERT(head_num_idx >= 2);
    x = z->inputs().at(head_num_idx)->expr().first;
    if (!helpers::IsConstant(x)) {
        return 0;
    }
    auto var_num_head = z->inputs().at(head_num_idx);
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
        EXPRP node_q, node_k, node_v;
        // whether transpose
        x = expr->inputs().at(0)->expr().first;
        if (!expr->get() || !helpers::IsTranspose(x)) {
            return false;
        }
        z = x;

        // whether reshape
        x = z->inputs().at(0)->expr().first;
        if (helpers::IsReshape(x)) {
            z = x;
            x = z->inputs().at(0)->expr().first;
        }

        // whether cast
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
        var_v = z->inputs().at(0);
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
        var_k = z->inputs().at(0);
        node_k = z->inputs().at(0)->expr().first;
        // whether mul(scale)
        if (helpers::IsBinaryMul(node_k)) {
            var_k = node_k->inputs().at(0);
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
        var_q = z->inputs().at(0);
        node_q = z->inputs().at(0)->expr().first;
        if (!helpers::IsMatMul(node_q)) {
            return false;
        }

        // QKV -> one source
        if (node_q->inputs().at(0)->expr().first != node_k->inputs().at(0)->expr().first || node_q->inputs().at(0)->expr().first != node_v->inputs().at(0)->expr().first) {
            return false;
        }
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
            // For target version < 2.8 , don't support attention
            return false;
        }

        if (expr->name().size() > 0) {
            MNN_PRINT("Fuse Original Self-Attention as %s\n", expr->name().c_str());
        }

        auto var_q_weight_info    = var_q_weight->getInfo();
        auto var_k_weight_info    = var_k_weight->getInfo();
        auto var_v_weight_info    = var_v_weight->getInfo();
        if (!var_q_weight_info || !var_k_weight_info || !var_v_weight_info || var_q_weight_info->size != var_k_weight_info->size || var_q_weight_info->size != var_v_weight_info->size) {
            return false;
        }
        /*
         query : [Batch, seqLen, headNum, headDim]
         key   : [Batch, seqLen, headNum, headDim]
         value : [Batch, seqLen, headNum, headDim]
         ouput : [Batch, seqLen, headNum * headDim]
         */
        var_q = _Reshape(var_q, {0, 0, mNumHeads, var_q_weight->getInfo()->dim[1] / mNumHeads});
        var_k = _Reshape(var_k, {0, 0, mNumHeads, var_q_weight->getInfo()->dim[1] / mNumHeads});
        var_v = _Reshape(var_v, {0, 0, mNumHeads, var_q_weight->getInfo()->dim[1] / mNumHeads});
        std::unique_ptr<MNN::AttentionParamT> param_attn(new MNN::AttentionParamT);
        param_attn->kv_cache = false;
        std::unique_ptr<OpT> attention(new OpT);
        attention->name       = "Attention" + expr->name();
        attention->type       = OpType_Attention;
        attention->main.type  = OpParameter_AttentionParam;
        attention->main.value = param_attn.release();
        auto attention_expr = Variable::create(Expr::create(attention.get(), {var_q, var_k, var_v}, 1));
        attention_expr->setName(expr->name());
        Expr::replace(expr, attention_expr->expr().first);

        return true /*modified*/;
    };

    TemplateMerge::getInstance("Merge").insertTemplate("FuseFmhaV2", match, fold);
}

class FuseSelfAttentionV2 {
public:
    FuseSelfAttentionV2();
private:
    VARP var_q, var_k, var_v;
    VARP var_q_weight, var_k_weight, var_v_weight;
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
        EXPRP node_q, node_k, node_v;
        // whether transpose
        x = expr->inputs().at(0)->expr().first;
        if (!expr->get() || !helpers::IsTranspose(x)) {
            return false;
        }
        z = x;

        // whether reshape
        x = z->inputs().at(0)->expr().first;
        if (helpers::IsReshape(x)) {
            z = x;
            x = z->inputs().at(0)->expr().first;
        }

        // whether Einsum/MatMul
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
        var_v = z->inputs().at(0);
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
        }

        // whether mul(scale)
        if (helpers::IsBinaryMul(x)) {
            x = x->inputs().at(0)->expr().first;
        }
        
        //whether matmul
        if (helpers::IsMatMul(x)) {
            z = x;
        } else {
            return false;
        }

        auto q_pre = z->inputs().at(0)->expr().first;
        auto k_pre = z->inputs().at(1)->expr().first;
        // whether mul(scale)
        if (helpers::IsBinaryMul(q_pre)) {
            q_pre = q_pre->inputs().at(0)->expr().first;
        }
        if (helpers::IsBinaryMul(k_pre)) {
            k_pre = k_pre->inputs().at(0)->expr().first;
        }

        z = GetFmhaV2BlockCommonNode(k_pre);
        if (z == nullptr) {
            return false;
        }

        if (mNumHeads != GetFmhaV2NumHeads(z)) {
            return false;
        }
        var_k = z->inputs().at(0);
        node_k = z->inputs().at(0)->expr().first;
        // whether mul(scale)
        if (helpers::IsBinaryMul(node_k)) {
            var_k = node_k->inputs().at(0);
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
        var_q = z->inputs().at(0);
        node_q = z->inputs().at(0)->expr().first;
        if (!helpers::IsMatMul(node_q)) {
            return false;
        }

        // QKV -> one source
        if (node_q->inputs().at(0)->expr().first != node_k->inputs().at(0)->expr().first || node_q->inputs().at(0)->expr().first != node_v->inputs().at(0)->expr().first) {
            return false;
        }
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

        if (expr->name().size() > 0) {
            MNN_PRINT("Fuse Original Self-Attention as %s\n", expr->name().c_str());
        }

        auto var_q_weight_info    = var_q_weight->getInfo();
        auto var_k_weight_info    = var_k_weight->getInfo();
        auto var_v_weight_info    = var_v_weight->getInfo();
        if (!var_q_weight_info || !var_k_weight_info || !var_v_weight_info || var_q_weight_info->size != var_k_weight_info->size || var_q_weight_info->size != var_v_weight_info->size) {
            return false;
        }

        /*
         query : [Batch, seqLen, headNum, headDim]
         key   : [Batch, seqLen, headNum, headDim]
         value : [Batch, seqLen, headNum, headDim]
         ouput : [Batch, seqLen, headNum * headDim]
         */
        var_q = _Reshape(var_q, {0, 0, mNumHeads, var_q_weight->getInfo()->dim[1] / mNumHeads});
        var_k = _Reshape(var_k, {0, 0, mNumHeads, var_q_weight->getInfo()->dim[1] / mNumHeads});
        var_v = _Reshape(var_v, {0, 0, mNumHeads, var_q_weight->getInfo()->dim[1] / mNumHeads});
        std::unique_ptr<MNN::AttentionParamT> param_attn(new MNN::AttentionParamT);
        param_attn->kv_cache = false;
        std::unique_ptr<OpT> attention(new OpT);
        attention->name       = "Attention" + expr->name();
        attention->type       = OpType_Attention;
        attention->main.type  = OpParameter_AttentionParam;
        attention->main.value = param_attn.release();
        auto attention_expr = Variable::create(Expr::create(attention.get(), {var_q, var_k, var_v}, 1));
        attention_expr->setName(expr->name());
        Expr::replace(expr, attention_expr->expr().first);

        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("FuseSelfAttentionV2", match, fold);
}

class FuseSelfAttentionV3 {
public:
    FuseSelfAttentionV3();
private:
    VARP var_qkv;
    VARP var_qkv_weight, var_qkv_bias;
    int mNumHeads;
};

FuseSelfAttentionV3::FuseSelfAttentionV3() {
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
        EXPRP node_q, node_k, node_v;
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
        
        if (helpers::IsSqueeze(v_pre)) {
            z = v_pre;
        } else {
            return false;
        }

        EXPRP node_split = z->inputs().at(0)->expr().first;
        if (!helpers::IsSlice(node_split)) {
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
        // whether mul(scale)
        if (helpers::IsBinaryMul(q_pre)) {
            q_pre = q_pre->inputs().at(0)->expr().first;
        }
        if (helpers::IsBinaryMul(k_pre)) {
            k_pre = k_pre->inputs().at(0)->expr().first;
        }

        if (helpers::IsSqueeze(q_pre)) {
            z = q_pre;
        } else {
            return false;
        }
        
        if(node_split != z->inputs().at(0)->expr().first) {
            return false;
        }
        
        if (helpers::IsTranspose(k_pre)) {
            z = k_pre;
        } else {
            return false;
        }
        x = z->inputs().at(0)->expr().first;
        if (helpers::IsSqueeze(x)) {
            z = x;
        } else {
            return false;
        }
        if(node_split != z->inputs().at(0)->expr().first) {
            return false;
        }
        
        // whether transpose
        x = node_split->inputs().at(0)->expr().first;
        if (!helpers::IsTranspose(x)) {
            return false;
        }
        z = x;
        
        // whether reshape
        x = z->inputs().at(0)->expr().first;
        if (!helpers::IsReshape(x)) {
            return false;
        }
        z = x;
        mNumHeads = GetFmhaV2NumHeads(z);

        // whether matmul
        x = z->inputs().at(0)->expr().first;
        if (!helpers::IsMatMul(x)) {
            return false;
        }
        EXPRP node_qkv = x;
        
        // whether transpose
        x = node_qkv->inputs().at(0)->expr().first;
        if (!helpers::IsTranspose(x)) {
            return false;
        }
        z = x;
        
        // whether reshape
        x = z->inputs().at(0)->expr().first;
        if (!helpers::IsReshape(x)) {
            return false;
        }
        z = x;
        var_qkv  = z->inputs().at(0);
        var_qkv_weight = node_qkv->inputs().at(1);
        if(node_qkv->inputs().size() > 2) {
            return false;
        }

        if(!helpers::IsConstant(var_qkv_weight->expr().first)) {
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

        if (expr->name().size() > 0) {
            MNN_PRINT("Fuse Original Self-Attention as %s\n", expr->name().c_str());
        }

        // FuseQKV_Weight -> Split
        auto var_qkv_weight_reshape = _Reshape(var_qkv_weight, {0, 3, -1});
        auto splitvar = _Split(var_qkv_weight_reshape, {3}, 1);
        auto var_q_weight = _Unsqueeze(_Reshape(splitvar[0], {0, -1}), {0});
        auto var_k_weight = _Unsqueeze(_Reshape(splitvar[1], {0, -1}), {0});
        auto var_v_weight = _Unsqueeze(_Reshape(splitvar[2], {0, -1}), {0});

        // [batch, inChannel, h, w] -> [batch, inChannel, seqLen]
        auto var_qkv_reshape = _Reshape(var_qkv, {0, 0, -1});

        // [batch, seqLen, headNum * headDim]
        auto output_q = _MatMul(var_qkv_reshape, var_q_weight, true, false);
        auto output_k = _MatMul(var_qkv_reshape, var_k_weight, true, false);
        auto output_v = _MatMul(var_qkv_reshape, var_v_weight, true, false);


        /*
         query : [Batch, seqLen, headNum, headDim]
         key   : [Batch, seqLen, headNum, headDim]
         value : [Batch, seqLen, headNum, headDim]
         ouput : [Batch, seqLen, headNum * headDim]
         */
        output_q = _Reshape(output_q, {0, 0, mNumHeads, var_q_weight->getInfo()->dim[1] / mNumHeads});
        output_k = _Reshape(output_k, {0, 0, mNumHeads, var_q_weight->getInfo()->dim[1] / mNumHeads});
        output_v = _Reshape(output_v, {0, 0, mNumHeads, var_q_weight->getInfo()->dim[1] / mNumHeads});
        std::unique_ptr<MNN::AttentionParamT> param_attn(new MNN::AttentionParamT);
        param_attn->kv_cache = false;
        std::unique_ptr<OpT> attention(new OpT);
        attention->name       = "Attention" + expr->name();
        attention->type       = OpType_Attention;
        attention->main.type  = OpParameter_AttentionParam;
        attention->main.value = param_attn.release();
        auto attention_expr = Variable::create(Expr::create(attention.get(), {output_q, output_k, output_v}, 1));
        attention_expr->setName(expr->name());
        Expr::replace(expr, attention_expr->expr().first);

        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("FuseSelfAttentionV3", match, fold);
}

static FuseFmhaV2 g_fuse_fmhaV2;
static FuseSelfAttentionV2 g_fuse_self_fmhaV2;
static FuseSelfAttentionV3 g_fuse_attention_v3;

} // namespace Express
} // namespace MNN
