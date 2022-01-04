//
//  OnnxNot.cpp
//  MNNConverter
//
//  Created by MNN on 2021/04/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxNotTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto input   = expr->inputs()[0];
        auto one    = _Scalar<int32_t>(1);
        auto newExpr = _Negative(input-one)->expr().first;
        newExpr->setName(expr->name());
        return newExpr;
    }
};

class OnnxNoneZeroTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto input   = expr->inputs()[0];
        auto mask = _NotEqual(input, _ZerosLike(input));
        std::unique_ptr<OpT> whereOp(new OpT);
        whereOp->type = OpType_Where;
        whereOp->main.type = OpParameter_Extra;
        whereOp->main.value = new ExtraT;
        auto whereExpr = Expr::create(whereOp.get(), {mask});
        auto whereVar = Variable::create(whereExpr);
        auto res = _Transpose(whereVar, {1, 0});
        res->setName(expr->name());
        return res->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Not",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxNotTransform));
    OnnxExtraManager::get()->insert("NonZero",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxNoneZeroTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
