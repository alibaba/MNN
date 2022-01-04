//
//  OnnxPrelu.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxPreluTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        MNN_THROW_CHECK(inputs.size() == 2, "Onnx Prelu Should have 2 inputs!");

        auto slope     = inputs[1];
        auto slopeInfo = slope->getInfo();
        auto slopeData = slope->readMap<float>();
        if (slopeInfo == nullptr || slopeData == nullptr) {
            auto k = _Select(_Less(inputs[0], _Scalar<float>(0)), slope, _Scalar<float>(1));
            auto res = _Multiply(inputs[0], k);
            res->setName(expr->outputName(0));
            return res->expr().first;
        }

        const int slopeSize = slopeInfo->size;

        std::unique_ptr<PReluT> preluParam(new PReluT);

        preluParam->slopeCount = slopeSize;

        preluParam->slope.resize(slopeSize);
        memcpy(preluParam->slope.data(), slopeData, slopeSize * sizeof(float));

        // prelu(input, slope) => mergedPrelu(input)
        std::unique_ptr<OpT> mergedOp(new OpT);
        mergedOp->name       = expr->name();
        mergedOp->type       = OpType_PReLU;
        mergedOp->main.type  = OpParameter_PRelu;
        mergedOp->main.value = preluParam.release();
        auto newExpr         = Expr::create(mergedOp.get(), {inputs[0]});
        newExpr->setName(expr->name());
        return newExpr;
    }
};

class OnnxCeluTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        float alpha = 1;
        auto attrs = expr->get()->main_as_Extra()->attr();
        if (attrs != nullptr) {
            for (const auto& attr : *attrs) {
                if (attr->key()->str() == "alpha") {
                    alpha = attr->f();
                }
            }
        }
        auto input = expr->inputs()[0];
        auto alphaVar = _Const(alpha);
        auto y = _Multiply(_Subtract(_Exp(_Divide(input, alphaVar)), _Const(1.0f)), alphaVar);
        auto res = _Select(_Less(input, _Const(0.0f)), y, input);
        auto newExpr = res->expr().first;
        newExpr->setName(expr->name());
        return newExpr;
    }
};

class OnnxThresholdedReluTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        float alpha = 1;
        auto attrs = expr->get()->main_as_Extra()->attr();
        if (attrs != nullptr) {
            for (const auto& attr : *attrs) {
                if (attr->key()->str() == "alpha") {
                    alpha = attr->f();
                }
            }
        }
        auto input = expr->inputs()[0];
        auto res = _Select(_Greater(input, _Const(alpha)), input, _Const(0.0f));
        auto newExpr = res->expr().first;
        newExpr->setName(expr->name());
        return newExpr;
    }
};

class OnnxShrinkTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        float bias = 0, lambd = 0.5;
        auto attrs = expr->get()->main_as_Extra()->attr();
        if (attrs != nullptr) {
            for (const auto& attr : *attrs) {
                if (attr->key()->str() == "bias") {
                    bias = attr->f();
                } else if (attr->key()->str() == "lambd") {
                    lambd = attr->f();
                }
            }
        }
        auto input = expr->inputs()[0];
        auto biasVar = _Const(bias);
        auto res = _Select(_Greater(input, _Const(lambd)), _Subtract(input, biasVar), // x-bias for x > lambd
                        _Select(_Less(input, _Const(-lambd)), _Add(input, biasVar), // x+bias for x < -lambd
                            _Const(0.0))); // 0 for otherwise
        auto newExpr = res->expr().first;
        newExpr->setName(expr->name());
        return newExpr;
    }
};

class OnnxTriluTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto shape = _Shape(inputs[0]), zero = _Scalar<int>(0), oneV = _Unsqueeze(_Scalar<int>(1), {0});
        auto H = _Slice(shape, _Unsqueeze(_Scalar<int>(-2), {0}), oneV), W = _Slice(shape, _Unsqueeze(_Scalar<int>(-1), {0}), oneV);
        auto rangeH = _Unsqueeze(_Range(zero, H, oneV), {1}), rangeW = _Unsqueeze(_Range(zero, W, oneV), {0});
        bool upper = true;
        auto attrs = expr->get()->main_as_Extra()->attr();
        if (attrs != nullptr) {
            for (const auto& attr : *attrs) {
                if (attr->key()->str() == "upper") {
                    upper = attr->i();
                }
            }
        }
        auto k = (inputs.size() == 2 ? inputs[1] : _Scalar<int>(0));
        auto mask = (upper ? _GreaterEqual(rangeW, rangeH + k) : _GreaterEqual(rangeH, rangeW - k));
        mask = _Reshape(mask, _Concat({_Fill(_Size(shape) - _Scalar<int>(2), oneV), _Shape(mask)}, 0));
        auto res = _Select(mask, inputs[0], zero);
        res->setName(expr->outputName(0));
        return res->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("PRelu", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxPreluTransform));
    OnnxExtraManager::get()->insert("Celu", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxCeluTransform));
    OnnxExtraManager::get()->insert("ThresholdedRelu", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxThresholdedReluTransform));
    OnnxExtraManager::get()->insert("Shrink", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxShrinkTransform));
    OnnxExtraManager::get()->insert("Trilu", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxTriluTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
