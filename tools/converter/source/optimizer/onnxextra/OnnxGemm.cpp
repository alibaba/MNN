//
//  OnnxGemm.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxGemmTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op = expr->get();
        bool transA = false;
        bool transB = false;
        float alpha = 1.0f;
        float beta  = 1.0f;

        auto extraParam    = op->main_as_Extra();
        const int attrSize = extraParam->attr()->size();
        for (int i = 0; i < attrSize; ++i) {
            auto attr       = extraParam->attr()->GetAs<Attribute>(i);
            const auto& key = attr->key()->str();
            if (key == "transA") {
                transA = attr->i() > 0;
                continue;
            }
            if (key == "transB") {
                transB = attr->i() > 0;
                continue;
            }
            if (key == "alpha") {
                alpha = attr->f();
                continue;
            }
            if (key == "beta") {
                beta = attr->f();
                continue;
            }
        }
        auto X = inputs[0];
        auto Y = inputs[1];
        auto Z = _MatMul(X, Y, transA, transB);
        if (1.0f != alpha) {
            Z = Z * _Scalar<float>(alpha);
        }
        if (inputs.size() > 2) {
            auto B = inputs[2];
            if (1.0f != beta) {
                B = B * _Scalar<float>(beta);
            }
            Z = Z + B;
        }
        Z->setName(expr->name());
        
        return Z->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Gemm", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxGemmTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
