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
static VARP _MatMul_Int8(VARP a, VARP b, bool tranposeA, bool tranposeB, VARP scaleA, VARP zeroA, VARP scaleB, VARP zeroB, VARP ScaleOut, VARP ScaleZero, VARP bias = nullptr) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type                   = OpParameter_MatMul;
    op->type                        = OpType_MatMul;
    op->main.value                  = new MatMulT;
    op->main.AsMatMul()->transposeA = tranposeA;
    op->main.AsMatMul()->transposeB = tranposeB;
    return (Variable::create(Expr::create(op.get(), {a, b, scaleA, zeroA, scaleB, zeroB, ScaleOut, ScaleZero, bias})));
}

class OnnxGemmTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        bool transA = false;
        bool transB = false;
        float alpha = 1.0f;
        float beta  = 1.0f;
        bool op8 = false;

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
        auto x_expr = X->expr().first;
        auto y_expr = Y->expr().first;
        auto Z = _MatMul(X, Y, transA, transB);
        if (x_expr->get() && y_expr->get() && x_expr->get()->type() == OpType_Int8ToFloat && y_expr->get()->type() == OpType_Int8ToFloat) {
            // input quant info
            auto y_int8 = y_expr->inputs().at(0);
            auto y_scale = y_expr->inputs().at(2);
            auto y_zero = y_expr->inputs().at(3);
            auto x_int8 = x_expr->inputs().at(0);
            auto x_scale = x_expr->inputs().at(2);
            auto x_zero = x_expr->inputs().at(3);
            // output quant info
            auto outputExpr = expr->outputs().front().lock();
            auto outputScaleVar = outputExpr->inputs()[1];
            auto outputZero = _Const(0.f);
            if (outputExpr->inputs().size() > 2 && outputExpr->inputs()[2]->getInfo()) {
                if (outputExpr->inputs()[2]->getInfo()->type.code == halide_type_int) {
                    outputZero = _Cast<float>(outputExpr->inputs()[2]);
                } else {
                    outputZero = _Cast<float>(outputExpr->inputs()[2]) - _Const(128.f);
                }
            }
            
            Z = _MatMul_Int8(X, y_int8, transA, transB, x_scale, x_zero, y_scale, y_zero, outputScaleVar, outputZero);
            if (inputs.size() > 2) {
                auto bias_expr = inputs[2]->expr().first;
                auto bias_int32 = bias_expr->inputs().at(1);
                Z = _MatMul_Int8(X, y_int8, transA, transB, x_scale, x_zero, y_scale, y_zero, outputScaleVar, outputZero, bias_int32);
            }
            Z->setName(expr->name());
            return Z->expr().first;
        }
        
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
