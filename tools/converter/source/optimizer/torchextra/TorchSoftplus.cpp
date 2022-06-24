//
//  TorchSoftplus.cpp
//  MNNConverter
//
//  Created by MNN on 2022/04/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TorchExtraManager.hpp"
#include "logkit.h"

namespace MNN {
namespace Express {

class TorchSoftplusTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        MNN_ASSERT(inputs.size() == 1);
        auto info = op->main_as_Extra();
        int beta = 1;
        if (nullptr != info->attr()) {
            for (int i = 0; i < info->attr()->size(); ++i) {
                const auto attr          = info->attr()->GetAs<Attribute>(i);
                const auto attributeName = attr->key()->str();
                if (attributeName == "beta") {
                    beta = attr->i();
                }
            }
        }
        VARP softplus;
        auto x = inputs[0];
        if (beta == 1) {
            softplus = _Softplus(x);
        } else {
            auto beta_ = _Const(beta);
            softplus = _Log(_Add(_Exp(x * beta_), _Const(1))) / beta_;
        }
        softplus->setName(opName);
        return softplus->expr().first;
    }
};

class TorchBitwiseNotTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        MNN_ASSERT(inputs.size() == 1);
        auto info = op->main_as_Extra();
        auto x = inputs[0];
        VARP bitwise_not = _Scalar<int>(1) - x;
        bitwise_not->setName(opName);
        return bitwise_not->expr().first;
    }
};

static auto gRegister = []() {
    TorchExtraManager::get()->insert("softplus", std::shared_ptr<TorchExtraManager::Transform>(new TorchSoftplusTransform));
    TorchExtraManager::get()->insert("bitwise_not", std::shared_ptr<TorchExtraManager::Transform>(new TorchBitwiseNotTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
