//
//  ClipTransform.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CaffeExtraManager.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {

class ClipTransform : public CaffeExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto inputs = expr->inputs();
        auto min = op->main_as_Extra()->attr()->GetAs<Attribute>(0)->f();
        auto max = op->main_as_Extra()->attr()->GetAs<Attribute>(1)->f();

        auto newVar = _Maximum(_Minimum(inputs[0], _Const(max)), _Const(min));
        return newVar->expr().first;
    }
};

static auto gRegister = []() {
    CaffeExtraManager::get()->insert("Clip", std::shared_ptr<CaffeExtraManager::Transform>(new ClipTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
