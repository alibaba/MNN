//
//  ExpTransform.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cmath>
#include "CaffeExtraManager.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {

class ExpTransform : public CaffeExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op     = expr->get();
        auto inputs = expr->inputs();
        auto base   = op->main_as_Extra()->attr()->GetAs<Attribute>(0)->f();
        auto scale  = op->main_as_Extra()->attr()->GetAs<Attribute>(1)->f();
        auto shift  = op->main_as_Extra()->attr()->GetAs<Attribute>(2)->f();

        auto exponent = _Add(_Multiply(inputs[0], _Const(scale)), _Const(shift));
        if (fabs(base - (-1)) < 1e-6) {
            base = exp(1);
        }
        auto newVar = _Pow(_Const(base), exponent);
        return newVar->expr().first;
    }
};

static auto gRegister = []() {
    CaffeExtraManager::get()->insert("Exp", std::shared_ptr<CaffeExtraManager::Transform>(new ExpTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
