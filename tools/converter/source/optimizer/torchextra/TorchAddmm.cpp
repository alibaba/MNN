//
//  TorchAddmm.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TorchExtraManager.hpp"

namespace MNN {
namespace Express {

class TorchAddmmTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        int alpha = 1;
        int beta  = 1;

        auto extraParam    = op->main_as_Extra();
        const int attrSize = extraParam->attr()->size();
        for (int i = 0; i < attrSize; ++i) {
            auto attr       = extraParam->attr()->GetAs<Attribute>(i);
            const auto& key = attr->key()->str();
            if (key == "alpha") {
                alpha = attr->i();
                continue;
            }
            if (key == "beta") {
                beta = attr->i();
                continue;
            }
        }
        // X = beta * A + alpha * (B x C)
        auto A = inputs[0];
        auto B = inputs[1];
        auto C = inputs[2];
        if (beta != 1) {
            A = A * _Scalar<int>(beta);
        }
        auto D = _MatMul(B, C);
        if (1 != alpha) {
            D = D * _Scalar<int>(alpha);
        }
        auto X = A + D;
        X->setName(expr->name());

        return X->expr().first;
    }
};

static auto gRegister = []() {
    TorchExtraManager::get()->insert("addmm", std::shared_ptr<TorchExtraManager::Transform>(new TorchAddmmTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
