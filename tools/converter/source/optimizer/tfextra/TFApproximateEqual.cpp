//
//  TFApproximateEqual.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNN_generated.h"
#include "TFExtraManager.hpp"

namespace MNN {
namespace Express {

class ApproximateEqualTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op     = expr->get();
        auto inputs = expr->inputs();

        float tolerance = 1e-5;
        auto extra      = op->main_as_Extra();
        if (nullptr != extra->attr()) {
            for (int i = 0; i < extra->attr()->size(); ++i) {
                auto attr = extra->attr()->GetAs<Attribute>(i);
                if (attr->key()->str() == "tolerance") {
                    tolerance = attr->f();
                }
            }
        }

        auto diff   = _Abs(_Subtract(inputs[0], inputs[1]));
        auto output = _Less(diff, _Const(tolerance));

        auto newExpr = output->expr().first;
        return newExpr;
    }
};

static auto gRegister = []() {
    TFExtraManager::get()->insert("ApproximateEqual",
                                  std::shared_ptr<TFExtraManager::Transform>(new ApproximateEqualTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
