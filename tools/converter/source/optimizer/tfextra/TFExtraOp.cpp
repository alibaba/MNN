//
//  TFExtraOp.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/*Treat Simple Extra Op*/

#include <MNN/expr/ExprCreator.hpp>
#include "MNN_generated.h"
#include "TFExtraManager.hpp"

namespace MNN {
namespace Express {
static VARP _Cast(VARP x, DataType src, DataType dst) {
    std::unique_ptr<OpT> castOp(new OpT);
    castOp->type                     = OpType_Cast;
    castOp->main.value               = new CastParamT;
    castOp->main.type                = OpParameter_CastParam;
    castOp->main.AsCastParam()->srcT = src;
    castOp->main.AsCastParam()->dstT = dst;
    return Variable::create(Expr::create(castOp.get(), {x}));
}
class LogSoftmaxTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op     = expr->get();
        auto inputs = expr->inputs();

        int axis = -1;
        if (nullptr != op->main_as_Extra()->attr()) {
            auto n = op->main_as_Extra()->attr()->size();
            for (int i = 0; i < n; ++i) {
                auto attr = op->main_as_Extra()->attr()->GetAs<Attribute>(i);
                if (attr->key()->str() == "axis") {
                    axis = attr->i();
                    break;
                }
            }
        }
        VARP x           = inputs[0];
        VARP max         = _ReduceMax(x, {axis}, true);
        VARP sum         = _ReduceSum(_Exp(x - max), {axis}, true);
        VARP newVar      = x - max - _Log(sum);
        return newVar->expr().first;
    }
};

class LogicalNotTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        // For bool tensor, the value is 1 or 0
        // 1 - value is equal to logcicalNot
        auto inputs = expr->inputs();
        auto one    = _Scalar<int32_t>(1);
        auto newVar = (one - inputs[0]);
        return newVar->expr().first;
    }
};

VARP _BroadcastToForward(VARP a, VARP shape) {
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_BroadcastTo;
    op->main.type  = OpParameter_Axis;
    auto param = new AxisT;
    param->axis = 1;
    op->main.value = param;
    return (Variable::create(Expr::create(std::move(op), {a, shape})));
}

class SelectTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs    = expr->inputs();
        MNN_ASSERT(inputs.size() == 3);
        auto cond   = inputs[0];
        auto tvalue = inputs[1];
        auto fvalue = inputs[1];
        auto newVar = _Select(_BroadcastToForward(inputs[0], _Shape(inputs[1])), inputs[1], inputs[2]);
        return newVar->expr().first;
    }
};

static auto gRegister = []() {
    TFExtraManager::get()->insert("LogicalNot", std::shared_ptr<TFExtraManager::Transform>(new LogicalNotTransform));
    TFExtraManager::get()->insert("LogSoftmax", std::shared_ptr<TFExtraManager::Transform>(new LogSoftmaxTransform));
    TFExtraManager::get()->insert("Select", std::shared_ptr<TFExtraManager::Transform>(new SelectTransform));
    return true;
}();
} // namespace Express
} // namespace MNN
