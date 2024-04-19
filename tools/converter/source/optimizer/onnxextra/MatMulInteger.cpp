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
        /*
        auto inputs = expr->inputs();
        auto x = inputs[0];
        auto y = inputs[1];
        y = _Cast<float>(y);
        auto offset = _Const(128.0f);

        auto x_int8 = inputs[0];
        auto x_fp32 = _Int8ToFloat(inputs[0], _Const(1.0));

        auto y_fp32 = y - offset;
        auto y_int8 = _Cast<int8_t>(y_fp32);

        auto x_zero_fp32 = inputs[2];
        auto y_shape = y->getInfo()->dim; // y:[K,N]
        auto y_zero = _Unsqueeze(_Cast<float>(inputs[3]), {0});
        auto y_zero_fp32 = y_zero - offset;
        auto y_reduce0 = _ReduceSum(y - y_zero, {0}, true); // y_:[1,N]
        auto x_reduce1 = _ReduceSum(x_fp32, {2}, true);
        auto z = _MatMul(x_int8, y_int8, false, false) - x_zero_fp32 * y_reduce0 - _MatMul(x_reduce1, y_zero_fp32);

        auto newExpr = z->expr().first;
        newExpr->setName(expr->name());
        return newExpr;
        */
        
       auto inputs = expr->inputs();
       std::unique_ptr<MNN::OpT> matmul(new MNN::OpT);
       matmul->type = OpType_MatMul;
       auto newExpr = MNN::Express::Expr::create(matmul.get(), inputs, 4);
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
