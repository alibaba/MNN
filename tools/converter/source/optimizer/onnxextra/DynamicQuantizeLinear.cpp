//
//  DynamicQuantizeLinear.cpp
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
// Ref from https://github.com/onnx/onnx/blob/main/docs/Operators.md#DynamicQuantizeLinear
class OnnxDynamicQuantizeLinearTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto x   = expr->inputs()[0];
        auto range = _Scalar<float>(1.0f/255.0f);
        auto maxX = _ReduceMax(x);
        auto minX = _ReduceMin(x);
        auto scale = (maxX - minX) * range;
        auto scaleReq = _Reciprocal(scale);
        // Qmin = 0
        auto interZero = _Negative(minX * scaleReq);
        auto zeroFloat = _Round(_Relu6(interZero, 0.0f, 255.0f));
        auto zero = _Cast<uint8_t>(zeroFloat);
        auto y = _Cast<uint8_t>(_Round(_Relu6(_Round(x * scaleReq) + zeroFloat, 0.0f, 255.0f)));
        std::unique_ptr<MNN::OpT> iden(new MNN::OpT);
        iden->type = OpType_Identity;
        

        auto newExpr = MNN::Express::Expr::create(iden.get(), {y, scale, zero}, 3);
        newExpr->setName(expr->name());
        for (int i=0; i<3; ++i) {
            auto v = MNN::Express::Variable::create(newExpr, i);
            v->setName(expr->outputName(i));
        }
        return newExpr;
    }
};


static auto gRegister = []() {
    OnnxExtraManager::get()->insert("DynamicQuantizeLinear",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxDynamicQuantizeLinearTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
