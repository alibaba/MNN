//
//  PowerTransform.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CaffeExtraManager.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {

class PowerTransform : public CaffeExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op     = expr->get();
        auto inputs = expr->inputs();
        auto scale  = op->main_as_Extra()->attr()->GetAs<Attribute>(0)->f();
        auto shift  = op->main_as_Extra()->attr()->GetAs<Attribute>(1)->f();
        auto power  = op->main_as_Extra()->attr()->GetAs<Attribute>(2)->f();

        auto base   = _Add(_Multiply(inputs[0], _Const(scale)), _Const(shift));
        auto newVar = _Pow(base, _Const(power));
        return newVar->expr().first;
    }
};

static auto gRegister = []() {
    CaffeExtraManager::get()->insert("Power", std::shared_ptr<CaffeExtraManager::Transform>(new PowerTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
