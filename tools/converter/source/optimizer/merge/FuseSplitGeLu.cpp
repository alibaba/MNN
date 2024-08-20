//
//  FuseSplitGeLu.cpp
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

class FuseSplitGeLu {
public:
    FuseSplitGeLu();
private:
    VARP split_gelu_input_0;
    VARP split_gelu_input_1;
    bool mHasPrefixAdd = false;
};

bool IsUnaryErf(EXPRP expr) {
    const Op* op = expr->get();
    if (op == nullptr || op->type() != OpType_UnaryOp) {
        return false;
    }
    return op->main_as_UnaryOp()->opType() == UnaryOpOperation_ERF;
}

FuseSplitGeLu::FuseSplitGeLu() {
    auto match = [this](EXPRP expr) -> bool {
        auto config = Global<modelConfig>::Get();
        if(!config->transformerFuse) {
            return false;
        }
        // mul
        if (!expr->get() || !helpers::IsBinaryMul(expr)) {
            return false;
        }
        EXPRP x, y, z;
        EXPRP add_1, add_2;
        EXPRP mul_1, mul_2, mul_3;
        
        // mul
        x = expr->inputs().at(0)->expr().first;
        y = expr->inputs().at(1)->expr().first;
        if (helpers::IsBinaryMul(x) && helpers::IsSlice(y)) {
            z = x;
            if(!helpers::IsConstant(y->inputs().at(1)->expr().first) || y->inputs().at(1)->readMap<int32_t>()[0] != 0) {
                return false;
            }
            if(!helpers::IsConstant(y->inputs().at(3)->expr().first) || y->inputs().at(3)->readMap<int32_t>()[0] != -1) {
                return false;
            }
        } else if (helpers::IsBinaryMul(y) && helpers::IsSlice(x)) {
            z = y;
            if(!helpers::IsConstant(x->inputs().at(1)->expr().first) || x->inputs().at(1)->readMap<int32_t>()[0] != 0) {
                return false;
            }
            if(!helpers::IsConstant(x->inputs().at(3)->expr().first) || x->inputs().at(3)->readMap<int32_t>()[0] != -1) {
                return false;
            }
        } else {
            return false;
        }
        
        // x * a
        x = z->inputs().at(0)->expr().first;
        y = z->inputs().at(1)->expr().first;
        if (helpers::IsBinaryMul(x) && helpers::IsConstant(y)) {
            z = x;
        } else if (helpers::IsBinaryMul(y) && helpers::IsConstant(x)) {
            z = y;
        } else {
            return false;
        }
        
        // mul
        x = z->inputs().at(0)->expr().first;
        y = z->inputs().at(1)->expr().first;
        if (helpers::IsBinaryAdd(x) && helpers::IsSlice(y)){
            z = x;
        } else if (helpers::IsBinaryAdd(y) && helpers::IsSlice(x)){
            z = y;
        } else {
            return false;
        }
        
        // z -> x + a
        x = z->inputs().at(0)->expr().first;
        y = z->inputs().at(1)->expr().first;
        if (IsUnaryErf(x) && helpers::IsConstant(y)){
            z = x;
        } else if (helpers::IsConstant(x) && IsUnaryErf(y)) {
            z = y;
        } else {
            return false;
        }
        
        // z -> erf
        x = z->inputs().at(0)->expr().first;
        if (helpers::IsBinaryRealDiv(x)){
            z = x;
        } else {
            return false;
        }
        // div (x / a)
        x = z->inputs().at(0)->expr().first;
        y = z->inputs().at(1)->expr().first;
        if (helpers::IsSlice(x) && helpers::IsConstant(y)){
            z = x;
        } else {
            return false;
        }
        // slice
        x = z->inputs().at(0)->expr().first;
        auto res = z;
        mHasPrefixAdd = false;
        if (helpers::IsBinaryAdd(x)) {
            z = x;
            x = z->inputs().at(0)->expr().first;
            y = z->inputs().at(1)->expr().first;
            if (helpers::IsConstant(x)){
                if(z->inputs().at(0)->getInfo()->dim.size() == 1) {
                    split_gelu_input_0 = z->inputs().at(1);
                    split_gelu_input_1 = z->inputs().at(0);
                } else {
                    split_gelu_input_0 = res->inputs().at(0);
                    return true;
                }
            } else if (helpers::IsConstant(y)){
                if(z->inputs().at(1)->getInfo()->dim.size() == 1) {
                    split_gelu_input_0 = z->inputs().at(0);
                    split_gelu_input_1 = z->inputs().at(1);
                } else {
                    split_gelu_input_0 = res->inputs().at(0);
                    return true;
                }
            } else {
                split_gelu_input_0 = res->inputs().at(0);
                return true;
            }
            mHasPrefixAdd = true;
        } else {
            split_gelu_input_0 = res->inputs().at(0);
        }
        return true;
    };

    auto fold = [this](EXPRP expr) -> bool {
        auto config = Global<modelConfig>::Get();
        auto version = config->targetVersion;
        
        if (version < 2.8f) {
            // For target version < 2.8 , don't support split_gelu
            return false;
        }
        std::unique_ptr<OpT> split_gelu_op(new OpT);
        split_gelu_op->name       = expr->name();
        split_gelu_op->type       = OpType_SplitGeLU;
        if (mHasPrefixAdd) {
            auto split_gelu_expr = Variable::create(Expr::create(split_gelu_op.get(), {split_gelu_input_0, split_gelu_input_1}, 1));
            split_gelu_expr->setName("SplitGeLU_" + expr->name());
            Expr::replace(expr, split_gelu_expr->expr().first);
        } else {
            auto split_gelu_expr = Variable::create(Expr::create(split_gelu_op.get(), { split_gelu_input_0 }, 1));
            split_gelu_expr->setName("SplitGeLU_" + expr->name());
            Expr::replace(expr, split_gelu_expr->expr().first);
        }
        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("FuseSplitGeLu", match, fold);
}

static FuseSplitGeLu g_fuse_splitgelu;

} // namespace Express
} // namespace MNN
