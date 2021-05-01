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
        randomUniform->type       = OpType_RandomUniform;
        randomUniform->main.type  = OpParameter_RandomUniform;
        auto param = new RandomUniformT;
        randomUniform->main.value = param;
        std::vector<int> outoutShape;
        for (int i = 0; i < info->attr()->size(); ++i) {
            const auto attr          = info->attr()->GetAs<Attribute>(i);
            const auto attributeName = attr->key()->str();
            if (attributeName == "shape") {
                if (nullptr != attr->list() && nullptr != attr->list()->i()) {
                    outoutShape.resize(attr->list()->i()->size());
                    ::memcpy(outoutShape.data(), attr->list()->i()->data(), outoutShape.size() * sizeof(int));
                }
            } else if (attributeName == "low") {
                param->low = attr->f();
            } else if (attributeName == "high") {
                param->high = attr->f();
            } else if (attributeName == "seed") {
                param->seed = attr->i();
            } else if (attributeName == "dtype") {
                param->type = static_cast<MNN::DataType>(attr->i());
            }
        }
        auto const_shape = _Const(static_cast<const void *>(outoutShape.data()), { static_cast<int>(outoutShape.size()) }, NCHW, halide_type_of<int>());
        auto newExpr = Expr::create(randomUniform.get(), {const_shape});
        newExpr->setName(expr->name());
        return newExpr;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("RandomUniform",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxRandomUniformTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
