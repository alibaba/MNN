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
        auto offset = _Const(128.0f);
        auto x_fp32 = x - offset;
        auto y_fp32 = y - offset;
        auto y_int8 = _Cast<int8_t>(y_fp32);
        auto x_int8 = _FloatToInt8(x_fp32,  _Scalar<float>(1.0f), -128, 127);
        auto z = _MatMul(x_int8, y_int8);
        if (inputs.size() > 2) {
            auto x_zero_fp32 = _Cast<float>(inputs[2]) - offset;
            auto x_zero_int8 = _Cast<int8_t>(x_zero_fp32);
            auto y_shape = y->getInfo()->dim; // y:[K,N]
            auto y_zero = _Unsqueeze(_Cast<float>(inputs[3]), {0});
            auto y_zero_fp32 = y_zero - offset;
            auto y_zero_1xN = _Cast<int32_t>(y_zero_fp32);
            int N = y_shape[1];
            auto y_reduce0 = _ReduceSum(y - y_zero, {0}, true); // y_:[1,N]
            auto x_reduce1 = _ReduceSum(x_fp32, {2}, true);
            z = _MatMul(x_int8, y_int8) - x_zero_fp32 * y_reduce0 - _MatMul(x_reduce1, y_zero_fp32);
        }
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
