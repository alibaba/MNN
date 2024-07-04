//
//  FuseGroupNorm.cpp
//  MNNConverter
//
//  Created by MNN on 2024/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <unordered_map>

#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"
#include "MergeHelpers.hpp"

namespace MNN {
namespace Express {

class FuseGroupNormWithSwish {
public:
    FuseGroupNormWithSwish();
protected:
    bool match_group_norm(EXPRP expr, bool testSwish);
    bool fold_group_norm(EXPRP expr, bool testSwish);
private:
    VARP group_norm_input_0;
    VARP group_norm_input_1;
    float mEpsilon;
    int mSwish = 0;
    bool mHasPrefixAdd = false;
    VARP gamma_var_;
    VARP beta_var_;
    
    EXPRP mGroupNorm;
    int mGroup;
};

bool IsSigmoid(EXPRP expr) {
    const Op* op = expr->get();
    if (op == nullptr) {
        return false;
    }
    if (op->type() == OpType_Sigmoid) {
        return true;
    }
    if (op->type() == OpType_UnaryOp && op->main_as_UnaryOp()->opType() == UnaryOpOperation_SIGMOID) {
        return true;
    }
    return false;
}
bool IsLayerNorm(EXPRP expr) {
    const Op* op = expr->get();
    if (op == nullptr) {
        return false;
    }
    if (op->type() == OpType_LayerNorm) {
        return true;
    }
    return false;
}
bool IsGroupNormNoSwish(EXPRP expr) {
    const Op* op = expr->get();
    if (op == nullptr) {
        return false;
    }
    if (op->type() == OpType_GroupNorm && op->main_as_GroupNorm()->bSwish() == 0) {
        return true;
    }
    return false;
}

bool FuseGroupNormWithSwish::match_group_norm(EXPRP expr, bool testSwish) {
    auto config = Global<modelConfig>::Get();
    if(!config->transformerFuse) {
        return false;
    }
    // mul
    if (!expr->get()) {
        return false;
    }
    EXPRP x, y, z;
    
    if (testSwish && helpers::IsBinaryMul(expr)) {
        // mul
        x = expr->inputs().at(0)->expr().first;
        y = expr->inputs().at(1)->expr().first;
        if (IsGroupNormNoSwish(x) && IsSigmoid(y) && (x == y->inputs().at(0)->expr().first)) {
            z = x;
        } else if (IsGroupNormNoSwish(y) && IsSigmoid(x) && (y == x->inputs().at(0)->expr().first)) {
            z = y;
        } else {
            return false;
        }
        mSwish = 1;
        mGroupNorm = z;
        return true;

    } else if (!testSwish && helpers::IsBinaryAdd(expr)) {
        z = expr;
    } else {
        return false;
    }

    // x * a
    x = z->inputs().at(0)->expr().first;
    y = z->inputs().at(1)->expr().first;
    if (helpers::IsBinaryMul(x) && helpers::IsConstant(y)) {
        beta_var_ = z->inputs().at(1);
        z = x;
    } else if (helpers::IsBinaryMul(y) && helpers::IsConstant(x)) {
        beta_var_ = z->inputs().at(0);
        z = y;
    } else {
        return false;
    }
    
    // reshape
    x = z->inputs().at(0)->expr().first;
    y = z->inputs().at(1)->expr().first;
    if (helpers::IsReshape(x) && helpers::IsConstant(y)) {
        gamma_var_ = z->inputs().at(1);
        z = x;
    } else if (helpers::IsReshape(y) && helpers::IsConstant(x)) {
        gamma_var_ = z->inputs().at(0);
        z = y;
    } else {
        return false;
    }
    
    // reshape -> LayerNorm
    x = z->inputs().at(0)->expr().first;
    if(helpers::IsReshape(x) && IsLayerNorm(x->inputs().at(0)->expr().first)) {
        z = x->inputs().at(0)->expr().first;
        mEpsilon = z->get()->main_as_LayerNorm()->epsilon();
    } else {
        return false;
    }
    
    // reshape -> Reshape
    x = z->inputs().at(0)->expr().first;
    if(helpers::IsReshape(x) && helpers::IsReshape(x->inputs().at(0)->expr().first)) {
        z = x->inputs().at(0)->expr().first;
    } else {
        return false;
    }
    x = z->inputs().at(0)->expr().first;
    
    auto var_reshape_group = z->inputs().at(1);
    mGroup = var_reshape_group->readMap<int32_t>()[1];
    if(mGroup < 1) {
        return false;
    }
    mHasPrefixAdd = false;
    if(helpers::IsBinaryAdd(x)) {
        auto add_ = x;
        x = add_->inputs().at(0)->expr().first;
        y = add_->inputs().at(1)->expr().first;
        if(helpers::IsConvolution(x) && helpers::IsUnsqueeze(y)) {
            x = y->inputs().at(0)->expr().first;
            if(helpers::IsUnsqueeze(x)) {
                mHasPrefixAdd = true;
                group_norm_input_0 = add_->inputs().at(0);
                group_norm_input_1 = x->inputs().at(0);
                return true;
            }
        }
    }
    group_norm_input_0 = z->inputs().at(0);
    return true;
}

bool FuseGroupNormWithSwish::fold_group_norm(EXPRP expr, bool testSwish) {
    
    auto config = Global<modelConfig>::Get();
    auto version = config->targetVersion;
    if (version < 2.8f) {
        // For target version < 2.8 , don't support group_norm
        return false;
    }

    std::unique_ptr<MNN::GroupNormT> group_norm(new MNN::GroupNormT);
    
    if(mSwish) {
        auto gn = mGroupNorm->get()->main_as_GroupNorm();
        group_norm->epsilon = gn->epsilon();
        group_norm->bSwish = 1;
        group_norm->group = gn->group();
        
        int size = gn->gamma()->size();
        group_norm->gamma.resize(size);
        group_norm->beta.resize(size);
        memcpy(group_norm->gamma.data(), gn->gamma()->data(), size * sizeof(float));
        memcpy(group_norm->beta.data(), gn->beta()->data(), size * sizeof(float));
        
        std::unique_ptr<OpT> group_norm_op(new OpT);
        group_norm_op->name       = mGroupNorm->name();
        group_norm_op->type       = OpType_GroupNorm;
        group_norm_op->main.type  = OpParameter_GroupNorm;
        group_norm_op->main.value = group_norm.release();
        
        auto group_norm_expr = Variable::create(Expr::create(group_norm_op.get(), mGroupNorm->inputs(), 1));
        group_norm_expr->setName("GroupNorm_" + expr->name());
        Expr::replace(expr, group_norm_expr->expr().first);
        return true /*modified*/;
    }
    
    group_norm->epsilon = mEpsilon;
    group_norm->bSwish = mSwish;
    group_norm->group = mGroup;
    
    auto* gamma_info   = gamma_var_->getInfo();
    auto* beta_info    = beta_var_->getInfo();
    const float* gamma = gamma_var_->readMap<float>();
    const float* beta  = beta_var_->readMap<float>();
    if (!gamma_info || !beta_info || !gamma || !beta || gamma_info->size != beta_info->size) {
        return false;
    }
    int size = gamma_info->size;
    group_norm->gamma.resize(size);
    group_norm->beta.resize(size);
    memcpy(group_norm->gamma.data(), gamma, size * sizeof(float));
    memcpy(group_norm->beta.data(), beta, size * sizeof(float));
    
    std::unique_ptr<OpT> group_norm_op(new OpT);
    group_norm_op->name       = expr->name();
    group_norm_op->type       = OpType_GroupNorm;
    group_norm_op->main.type  = OpParameter_GroupNorm;
    group_norm_op->main.value = group_norm.release();
    
    if(mHasPrefixAdd) {
        auto group_norm_expr = Variable::create(Expr::create(group_norm_op.get(), {group_norm_input_0, group_norm_input_1 }, 1));
        group_norm_expr->setName("GroupNorm_" + expr->name());
        Expr::replace(expr, group_norm_expr->expr().first);
    } else {
        auto group_norm_expr = Variable::create(Expr::create(group_norm_op.get(), { group_norm_input_0 }, 1));
        group_norm_expr->setName("GroupNorm_" + expr->name());
        Expr::replace(expr, group_norm_expr->expr().first);
    }
    return true /*modified*/;
}

FuseGroupNormWithSwish::FuseGroupNormWithSwish() {

    auto match_with_swish = [this](EXPRP expr) -> bool {
        return match_group_norm(expr, true);
    };
    
    auto fold_with_swish = [this](EXPRP expr) -> bool {
        return fold_group_norm(expr, true);
    };

    TemplateMerge::getInstance("Merge").insertTemplate("FuseGroupNormWithSwish", match_with_swish, fold_with_swish);
}

class FuseGroupNormNoSwish : public FuseGroupNormWithSwish {
public:
    FuseGroupNormNoSwish();
};

FuseGroupNormNoSwish::FuseGroupNormNoSwish() {

    auto match_no_swish = [this](EXPRP expr) -> bool {
        return match_group_norm(expr, false);
    };

    auto fold_no_swish = [this](EXPRP expr) -> bool {
        return fold_group_norm(expr, false);
    };

    TemplateMerge::getInstance("Merge").insertTemplate("FuseGroupNormNoSwish", match_no_swish, fold_no_swish);
}

static FuseGroupNormWithSwish g_fuse_groupnorm_with_swish;
static FuseGroupNormNoSwish g_fuse_groupnorm_no_swish;

} // namespace Express
} // namespace MNN
