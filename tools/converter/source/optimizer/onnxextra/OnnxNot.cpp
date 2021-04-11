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

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Not",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxNotTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
