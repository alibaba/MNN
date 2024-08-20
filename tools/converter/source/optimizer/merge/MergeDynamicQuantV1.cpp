//
//  MergeDynamicQuantV1.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"
#include "MergeHelpers.hpp"

namespace MNN {
namespace Express {

static bool IsDynamicQuant(EXPRP expr) {
    const Op* op = expr->get();
    if (op && op->type() == OpType_DynamicQuant) {
        return true;
    }
    return false;
}

static VARPS _DynamicQuant(VARP x) {
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_DynamicQuant;
    op->main.type = OpParameter_NONE;
    op->main.value = nullptr;
    EXPRP expr = Expr::create(std::move(op), {x}, 3);
    return { Variable::create(expr, 0), Variable::create(expr, 1), Variable::create(expr, 2) };
}

static VARP _MatMul(VARP a, VARP b, VARP scale, bool tranposeA, bool tranposeB) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type                   = OpParameter_MatMul;
    op->type                        = OpType_MatMul;
    op->main.value                  = new MatMulT;
    op->main.AsMatMul()->transposeA = tranposeA;
    op->main.AsMatMul()->transposeB = tranposeB;
    return (Variable::create(Expr::create(op.get(), {a, b, scale})));
}

class DynamicQuantMatmulV1 {
public:
    DynamicQuantMatmulV1();

private:
    VARP mBias;
    VARP mWeightScale;
    VARP mWeight;
    VARP mWeightZero;
    VARP mDynamicQuantInput;
    VARPS mDynamicQuantOutputs;
};

DynamicQuantMatmulV1::DynamicQuantMatmulV1() {
    auto match = [this](EXPRP expr) -> bool {
        // check convInt8
        if (nullptr == expr->get()) {
            return false;
        }
        if (!helpers::IsBinaryAdd(expr)) {
            return false;
        }
        
        mBias = expr->inputs().at(0);
        auto floatMatmulResult = expr->inputs().at(1);
        
        if (!helpers::IsConstant(mBias->expr().first) && !helpers::IsConstant(floatMatmulResult->expr().first)) {
            return false;
        }
        
        if (!helpers::IsBinaryOp(mBias->expr().first) && !helpers::IsBinaryOp(floatMatmulResult->expr().first)) {
            return false;
        }
        if (helpers::IsBinaryOp(mBias->expr().first)) {
            mBias = expr->inputs().at(1);
            floatMatmulResult = expr->inputs().at(0);
        }
        if(!helpers::IsBinaryMul(floatMatmulResult->expr().first)) {
            return false;
        }
        
        auto matmulVar = floatMatmulResult->expr().first->inputs()[0];
        auto scaleVar = floatMatmulResult->expr().first->inputs()[1];
        // two branch: 1. Matmul->cast->Mul; 2. Matmul->Mul
        // first, matmulVar or scaleVar is cast
        if (helpers::IsCast(matmulVar->expr().first)) {
            auto castVar = matmulVar;
            matmulVar = castVar->expr().first->inputs()[0];
        }
        if (helpers::IsCast(scaleVar->expr().first)) {
                auto castVar = scaleVar;
                scaleVar = matmulVar;
                matmulVar = castVar->expr().first->inputs()[0];
        }

        if (!helpers::IsMatMul(matmulVar->expr().first) && !helpers::IsMatMul(scaleVar->expr().first)) {
            return false;
        }
        if (!helpers::IsBinaryOp(matmulVar->expr().first) && !helpers::IsBinaryOp(scaleVar->expr().first)) {
            return false;
        }
        if (helpers::IsBinaryOp(matmulVar->expr().first)) {
            matmulVar = floatMatmulResult->expr().first->inputs()[1];
            scaleVar = floatMatmulResult->expr().first->inputs()[0];
        }
        if (!helpers::IsBinaryMul(scaleVar->expr().first)) {
            return false;
        }
        
        mWeightScale = scaleVar->expr().first->inputs()[1];
        auto inputScale = scaleVar->expr().first->inputs()[0];
        if (!helpers::IsConstant(mWeightScale->expr().first) && !helpers::IsConstant(inputScale->expr().first)) {
            return false;
        }
        if (!IsDynamicQuant(mWeightScale->expr().first) && !IsDynamicQuant(inputScale->expr().first)) {
            return false;
        }
        if (helpers::IsConstant(inputScale->expr().first)) {
            mWeightScale = scaleVar->expr().first->inputs()[0];
            inputScale = scaleVar->expr().first->inputs()[1];
        }

        if (matmulVar->expr().first->inputs().size() != 4) {
            return false;
        }

        mWeight = matmulVar->expr().first->inputs()[1];
        mWeightZero = matmulVar->expr().first->inputs()[3];
        auto input = matmulVar->expr().first->inputs()[0];
        auto inputZero = matmulVar->expr().first->inputs()[2];
        if (!helpers::IsConstant(mWeight->expr().first) || !helpers::IsConstant(mWeightZero->expr().first)) {
            return false;
        }
        if (!IsDynamicQuant(input->expr().first) || !IsDynamicQuant(inputZero->expr().first)) {
            return false;
        }
        
        if (input->expr().first != inputZero->expr().first) {
            return false;
        }
        if (input->expr().first != inputScale->expr().first) {
            return false;
        }
        auto dynamicQuantExpr = input->expr().first;
        mDynamicQuantInput = dynamicQuantExpr->inputs().at(0);
        mDynamicQuantOutputs = _DynamicQuant(mDynamicQuantInput);
        
        return true;
    };
    auto transform = [this](EXPRP expr) -> bool {
        auto y = mWeight;
        y = _Cast<float>(y);
        auto offset = _Const(128.0f);

        auto x_int8 = mDynamicQuantOutputs[0];
        auto x_fp32 = _Int8ToFloat(x_int8, _Const(1.0));

        auto y_fp32 = y - offset;
        auto y_int8 = _Cast<int8_t>(y_fp32);

        auto x_zero_fp32 = mDynamicQuantOutputs[2];
        auto y_shape = y->getInfo()->dim; // y:[K,N]
        auto y_zero = _Unsqueeze(_Cast<float>(mWeightZero), {0});
        auto y_zero_fp32 = y_zero - offset;
        auto y_reduce0 = _ReduceSum(y - y_zero, {0}, true); // y_:[1,N]
        auto x_reduce1 = _ReduceSum(x_fp32, {2}, true);
//        auto z = _MatMul(x_int8, y_int8) - x_zero_fp32 * y_reduce0 - _MatMul(x_reduce1, y_zero_fp32);
//
//        auto newExpr = z->expr().first;
//        newExpr->setName(expr->name());
//        return newExpr;
        // first term
        auto convInt8 = _MatMul(x_int8, y_int8, mWeightScale, false, false);
        // second term
        auto y_reduce_mul_yscale = y_reduce0 * mWeightScale;
        auto sub1 = x_zero_fp32 * y_reduce_mul_yscale;
        // third term
        auto y_zero_fp32_mul_yscale = y_zero_fp32 * mWeightScale;
        auto sub2 = _MatMul(x_reduce1, y_zero_fp32_mul_yscale);
        auto z_sub_bias = convInt8 - sub2;
        auto z_sub_xzero = sub1 * mDynamicQuantOutputs[1] - mBias;
        auto z = z_sub_bias * mDynamicQuantOutputs[1] - z_sub_xzero;
//        z = z * mDynamicQuantOutputs[1] + mBias;
        
        auto newExpr = z->expr().first;
        newExpr->setName(expr->name());
        Expr::replace(expr, newExpr);
        return true;
    };

   TemplateMerge::getInstance("Merge").insertTemplate("DynamicQuantMatMulInteger", match, transform, PASS_PRIORITY_HIGH);
    
}
static DynamicQuantMatmulV1 g_dynamic_quant_matmul_v1;

} // namespace Express
} // namespace MNN
