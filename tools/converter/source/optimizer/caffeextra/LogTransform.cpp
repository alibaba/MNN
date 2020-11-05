//
//  LogTransform.cpp
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

class LogTransform : public CaffeExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op     = expr->get();
        auto inputs = expr->inputs();
        auto base   = op->main_as_Extra()->attr()->GetAs<Attribute>(0)->f();
        auto scale  = op->main_as_Extra()->attr()->GetAs<Attribute>(1)->f();
        auto shift  = op->main_as_Extra()->attr()->GetAs<Attribute>(2)->f();

        auto exponent = _Add(_Multiply(inputs[0], _Const(scale)), _Const(shift));
        if (fabs(base - (-1)) < 1e-6) { // base == -1, which means natural base
            return exponent->expr().first;
        }
        auto newVar = _Divide(_Log(exponent), _Log(_Const(base)));
        return newVar->expr().first;
    }
};

static auto gRegister = []() {
    CaffeExtraManager::get()->insert("Log", std::shared_ptr<CaffeExtraManager::Transform>(new LogTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
