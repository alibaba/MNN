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
        float alpha = 0.2f;
        float beta  = 0.5f;
        auto op     = expr->get();
        auto extra  = op->main_as_Extra();
        MNN_ASSERT(nullptr != extra);
        auto attr = extra->attr();
        if (nullptr != attr) {
            for (int i = 0; i < attr->size(); ++i) {
                auto att = attr->GetAs<Attribute>(i);
                auto key = att->key()->str();
                if ("alpha" == key) {
                    alpha = att->f();
                }
                if ("beta" == key) {
                    beta = att->f();
                }
            }
        }
        auto input = expr->inputs()[0];
        auto res   = _Add(input * _Const(alpha, {}, NCHW), _Const(beta, {}, NCHW));
        res        = _Relu6(res, 0.0f, 1.0f);
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
