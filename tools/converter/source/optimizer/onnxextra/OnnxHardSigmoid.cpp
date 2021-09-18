//
//  OnnxHardSigmoid.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxHardSigmoidTransform : public OnnxExtraManager::Transform {
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
    OnnxExtraManager::get()->insert("HardSigmoid",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxHardSigmoidTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
