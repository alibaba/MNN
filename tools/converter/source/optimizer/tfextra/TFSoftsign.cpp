//
//  TFSoftsign.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNN_generated.h"
#include "TFExtraManager.hpp"

namespace MNN {
namespace Express {

class SoftsignTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto input   = expr->inputs()[0];
        auto newExpr = _Softsign(input)->expr().first;
        return newExpr;
    }
};

static auto gRegister = []() {
    TFExtraManager::get()->insert("Softsign", std::shared_ptr<TFExtraManager::Transform>(new SoftsignTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
