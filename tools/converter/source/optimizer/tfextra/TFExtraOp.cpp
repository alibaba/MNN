//
//  TFExtraOp.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/*Treat Simple Extra Op*/

#include "TFExtraManager.hpp"
#include "MNN_generated.h"
#include <MNN/expr/ExprCreator.hpp>

namespace MNN {
namespace Express {
static VARP _Cast(VARP x, DataType src, DataType dst) {
    std::unique_ptr<OpT> castOp(new OpT);
    castOp->type = OpType_Cast;
    castOp->main.value = new CastParamT;
    castOp->main.type = OpParameter_CastParam;
    castOp->main.AsCastParam()->srcT = src;
    castOp->main.AsCastParam()->dstT = dst;
    return Variable::create(Expr::create(castOp.get(), {x}));
}
class LogSoftmaxTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto inputs = expr->inputs();

        int axis = -1;
        if (nullptr != op->main_as_Extra()->attr()) {
            auto n = op->main_as_Extra()->attr()->size();
            for (int i=0; i<n; ++i) {
                auto attr = op->main_as_Extra()->attr()->GetAs<Attribute>(i);
                if (attr->key()->str() == "axis") {
                    axis = attr->i();
                    break;
                }
            }
        }
        auto newVar = _Log(_Softmax(inputs[0], axis));
        return newVar->expr().first;
    }
};

class LogicalNotTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto one = _Const(-1.0f);
        auto floatCast = _Cast(inputs[0], DataType_DT_BOOL, DataType_DT_FLOAT);
        auto floatCompute = _Negative(_Add(floatCast, one));
        auto newVar = _Cast(floatCompute, DataType_DT_FLOAT, DataType_DT_BOOL);
        return newVar->expr().first;
    }
};
static auto gRegister = []() {
    TFExtraManager::get()->insert("LogicalNot", std::shared_ptr<TFExtraManager::Transform>(new LogicalNotTransform));
    TFExtraManager::get()->insert("LogSoftmax", std::shared_ptr<TFExtraManager::Transform>(new LogSoftmaxTransform));
    return true;
}();
}
} // namespace MNN
