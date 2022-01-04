//
//  OnnxRandomUniform.cpp
//  MNNConverter
//
//  Created by MNN on 2021/04/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxRandomUniformTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op   = expr->get();
        auto info = op->main_as_Extra();
        std::unique_ptr<OpT> randomUniform(new OpT);
        randomUniform->name       = expr->name();
        if (info->type()->str() == "RandomUniform" || info->type()->str() == "RandomUniformLike") {
            randomUniform->type       = OpType_RandomUniform;
        } else {
            randomUniform->type       = OpType_RandomNormal;
        }
        randomUniform->main.type  = OpParameter_RandomUniform;
        auto param = new RandomUniformT;
        randomUniform->main.value = param;
        std::vector<int> outoutShape;
        bool hasShape = false;
        bool hasType = false;
        if (nullptr != info->attr()) {
            for (int i = 0; i < info->attr()->size(); ++i) {
                const auto attr          = info->attr()->GetAs<Attribute>(i);
                const auto attributeName = attr->key()->str();
                if (attributeName == "shape") {
                    if (nullptr != attr->list() && nullptr != attr->list()->i()) {
                        outoutShape.resize(attr->list()->i()->size());
                        ::memcpy(outoutShape.data(), attr->list()->i()->data(), outoutShape.size() * sizeof(int));
                    }
                    hasShape = true;
                } else if (attributeName == "low" || attributeName == "mean") {
                    param->low = attr->f();
                } else if (attributeName == "high" || attributeName == "scale") {
                    param->high = attr->f();
                } else if (attributeName == "seed") {
                    param->seed = attr->i();
                } else if (attributeName == "dtype") {
                    param->type = static_cast<MNN::DataType>(attr->i());
                    hasType = true;
                }
            }
        }
        EXPRP newExpr;
        if (hasShape) {
            auto const_shape = _Const(static_cast<const void *>(outoutShape.data()), { static_cast<int>(outoutShape.size()) }, NCHW, halide_type_of<int>());
            newExpr = Expr::create(randomUniform.get(), {const_shape});
        } else {
            if (!hasType) {
                auto info = expr->inputs()[0]->getInfo();
                if (nullptr == info) {
                    return nullptr;
                }
                if (info->type.code == halide_type_float) {
                    param->type = DataType_DT_FLOAT;
                } else if (info->type.code == halide_type_int) {
                    if (info->type.bytes() == 1) {
                        param->type = DataType_DT_INT8;
                    } else {
                        param->type = DataType_DT_INT32;
                    }
                } else if (info->type.code == halide_type_uint) {
                    if (info->type.bytes() == 1) {
                        param->type = DataType_DT_UINT8;
                    }
                }
            }
            newExpr = Expr::create(randomUniform.get(), {_Shape(expr->inputs()[0], NCHW)});
        }
        newExpr->setName(expr->name());
        return newExpr;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("RandomUniform",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxRandomUniformTransform));
    OnnxExtraManager::get()->insert("RandomUniformLike",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxRandomUniformTransform));
    OnnxExtraManager::get()->insert("RandomNormal",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxRandomUniformTransform));
    OnnxExtraManager::get()->insert("RandomNormalLike",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxRandomUniformTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
