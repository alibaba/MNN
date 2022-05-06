//
//  TorchHardSigmoid.cpp
//  MNNConverter
//
//  Created by MNN on 2021/10/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TorchExtraManager.hpp"
#include "logkit.h"

namespace MNN {
namespace Express {

class TorchHardSigmoidTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op     = expr->get();
        auto input = expr->inputs()[0];
        auto res   = _Add(input, _Const(3, {}, NCHW));
        res        = _Relu6(res, 0.0f, 6.0f) / _Const(6, {}, NCHW);
        res->setName(expr->name());
        return res->expr().first;
    }
};

static auto gRegister = []() {
    TorchExtraManager::get()->insert("hardsigmoid", std::shared_ptr<TorchExtraManager::Transform>(new TorchHardSigmoidTransform));
    return true;
}();
} // namespace Express
} // namespace MNN
