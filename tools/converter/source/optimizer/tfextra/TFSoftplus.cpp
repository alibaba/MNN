//
//  TFSoftplus.cpp
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

class SoftplusTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto input   = expr->inputs()[0];
        auto newExpr = _Softplus(input)->expr().first;
        return newExpr;
    }
};

static auto gRegister = []() {
    TFExtraManager::get()->insert("Softplus", std::shared_ptr<TFExtraManager::Transform>(new SoftplusTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
