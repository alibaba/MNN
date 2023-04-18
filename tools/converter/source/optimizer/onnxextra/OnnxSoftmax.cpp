//
//  OnnxSoftmax.cpp
//  MNNConverter
//
//  Created by MNN on 2023/03/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

VARP _ConstInt(int x) {
    return _Const(&x, {1}, NHWC, halide_type_of<int>());
}

class OnnxSoftmaxTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);
        int axis = -1, opsetVersion = 13;
        auto attrs  = op->main_as_Extra()->attr();
        if (nullptr != attrs) {
            for (int i = 0; i < attrs->size(); ++i) {
                auto attr = attrs->GetAs<Attribute>(i);
                if (attr->key()->str() == "axis") {
                    axis = attr->i();
                }
                if (attr->key()->str() == "onnx_opset_version") {
                    opsetVersion = attr->i();
                }
            }
        }
        auto input = expr->inputs()[0];
        if (opsetVersion >= 13 || axis == -1) {
            auto newExpr = _Softmax(input, axis)->expr().first;
            newExpr->setName(expr->name());
            return newExpr;
        }
        auto shape = _Shape(input, true);
        auto newShape = _Stack({_ReduceProd(_Slice(shape, _ConstInt(0), _ConstInt(axis))), _ReduceProd(_Slice(shape, _ConstInt(axis), _ConstInt(-1)))});
        input = _Reshape(input, newShape);
        auto output = _Softmax(input, -1);
        auto newExpr = _Reshape(output, shape)->expr().first;
        newExpr->setName(expr->name());
        return newExpr;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Softmax",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxSoftmaxTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
