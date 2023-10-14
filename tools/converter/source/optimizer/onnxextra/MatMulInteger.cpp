//
//  MatMulInteger.cpp
//  MNNConverter
//
//  Created by MNN on 2023/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {
// Ref from https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMulInteger
// Use float instead of uint8 to complete it
class OnnxMatMulIntegerTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto x = inputs[0];
        auto y = inputs[1];
        x = _Cast<float>(x);
        y = _Cast<float>(y);
        if (inputs.size() > 2) {
            x = x - _Cast<float>(inputs[2]);
            y = y - _Cast<float>(inputs[3]);
        }
        auto z = _MatMul(x, y);
        auto zInt = _Cast<int32_t>(z);
        auto newExpr = zInt->expr().first;
        newExpr->setName(expr->name());
        return newExpr;
    }
};


static auto gRegister = []() {
    OnnxExtraManager::get()->insert("MatMulInteger",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxMatMulIntegerTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
