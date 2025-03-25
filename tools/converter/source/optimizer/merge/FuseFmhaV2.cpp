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
        // [Batch, seqLen, headNum * headDim]
        auto output_q = _MatMul(fmha_v2_input_, var_q_weight);
        auto info    = fmha_v2_input_->getInfo();
        // [Batch, seqLen, headNum, headDim]
        output_q = _Reshape(output_q, {0, 0, mNumHeads, var_q_weight->getInfo()->dim[1] / mNumHeads});
        
        auto output_k = _MatMul(fmha_v2_input_, var_k_weight);
        output_k = _Reshape(output_k, {0, 0, mNumHeads, var_q_weight->getInfo()->dim[1] / mNumHeads});
        
        auto output_v = _MatMul(fmha_v2_input_, var_v_weight);
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
        // [Batch, seqLen, headNum * headDim]
        auto output_q = _MatMul(fmha_v2_input_, var_q_weight);
        // [Batch, seqLen, headNum, headDim]
        output_q = _Reshape(output_q, {0, 0, mNumHeads, var_q_weight->getInfo()->dim[1] / mNumHeads});
        
        auto output_k = _MatMul(fmha_v2_input_, var_k_weight);
        output_k = _Reshape(output_k, {0, 0, mNumHeads, var_q_weight->getInfo()->dim[1] / mNumHeads});
        
        auto output_v = _MatMul(fmha_v2_input_, var_v_weight);
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
    TemplateMerge::getInstance("Merge").insertTemplate("FuseSelfAttentionV2", match, fold);
}
    
static FuseFmhaV2 g_fuse_fmhaV2;
static FuseSelfAttentionV2 g_fuse_self_fmhaV2;

} // namespace Express
} // namespace MNN
