//
// MergeDynamicQuantV2.cpp
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

static bool IsMatMulInteger(EXPRP expr) {
    const Op* op = expr->get();
    if (op && op->type() && op->type() == OpType_Extra && op->main_as_Extra() && op->main_as_Extra()->type()->str() == "MatMulInteger") {
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

class DynamicQuantMatmulV2 {
public:
    DynamicQuantMatmulV2();

private:
    VARP mWeightScale;
    VARP mWeight;
    VARP mWeightZero;
    VARP mDynamicQuantInput;
    VARPS mDynamicQuantOutputs;
};

DynamicQuantMatmulV2::DynamicQuantMatmulV2() {
    auto match = [this](EXPRP expr) -> bool {
        // dynamic->matmulInteger->mul without bias
        if (nullptr == expr->get()) {
            return false;
        }
        if (!helpers::IsBinaryMul(expr)) {
            return false;
        }
        
        auto matmulVar = expr->inputs()[0];
        auto scaleVar = expr->inputs()[1];
        // two branch: 1. Matmul->cast->Mul; 2. Matmul->Mul
        // first, matmulv or scaleVar is cast
        if (helpers::IsCast(matmulVar->expr().first)) {
            auto castVar = matmulVar;
            matmulVar = castVar->expr().first->inputs()[0];
        }
        if (helpers::IsCast(scaleVar->expr().first)) {
            auto castVar = scaleVar;
            scaleVar = matmulVar;
            matmulVar = castVar->expr().first->inputs()[0];
        }
        if (!IsMatMulInteger(matmulVar->expr().first) && !IsMatMulInteger(scaleVar->expr().first)) {
            return false;
        }
        if (!helpers::IsBinaryOp(matmulVar->expr().first) && !helpers::IsBinaryOp(scaleVar->expr().first)) {
            return false;
        }
        if (helpers::IsBinaryOp(matmulVar->expr().first)) {
            auto temp = matmulVar;
            matmulVar = scaleVar;
            scaleVar = temp;
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
        auto dynamicQuantVar = input->expr().first;
        mDynamicQuantInput = dynamicQuantVar->inputs().at(0);
        mDynamicQuantOutputs = _DynamicQuant(mDynamicQuantInput);
        
        return true;
    };
    auto transform = [this](EXPRP expr) -> bool {
        auto y = mWeight;
        if (mWeight->getInfo() && mWeight->getInfo()->dim.size() != 2) {
            MNN_ERROR("!!!! Error: Do not support!\n");
            return false;
        }
        if (mWeightScale->getInfo() && mWeightZero->getInfo() && mWeightScale->getInfo()->dim.size() != mWeightZero->getInfo()->dim.size()) {
            MNN_ERROR("!!!! Error: Do not support!\n");
            return false;
        }
        y = _Cast<float>(y);
        auto offset = _Const(128.0f);
        if (mWeight->getInfo() && mWeight->getInfo()->type.code == halide_type_int && mWeight->getInfo()->type.bits == 8) {
            offset = _Const(0.f);
        }

        auto x_int8 = mDynamicQuantOutputs[0];
        auto x_fp32 = _Int8ToFloat(x_int8, _Const(1.0));

        auto y_fp32 = y - offset;
        auto y_int8 = _Cast<int8_t>(y_fp32);

        auto x_zero_fp32 = mDynamicQuantOutputs[2];
        auto y_shape = y->getInfo()->dim; // y:[K,N]
        auto y_zero = _Cast<float>(mWeightZero);
        auto y_zero_fp32 = y_zero - offset;
        auto y_reduce0 = _ReduceSum(y - y_zero, {0}, true); // y_:[1,N]
        auto x_reduce1 = _ReduceSum(x_fp32, {-1}, true);    // x_:[M,1]

        // first term
        auto yscale = mWeightScale;
        if (mWeightScale->getInfo() && mWeightScale->getInfo()->dim.size() == 0) {
            std::vector<float> _scale(y_reduce0->getInfo()->dim[1], mWeightScale->readMap<float>()[0]);
            yscale = _Const(_scale.data(), {(int)_scale.size()}, NHWC, halide_type_of<float>() );
        }
        auto convInt8 = _MatMul(x_int8, y_int8, yscale, false, false);
        // second term
        auto y_reduce_mul_yscale = y_reduce0 * mWeightScale;
        auto sub1 = x_zero_fp32 * y_reduce_mul_yscale;
        // third term
        auto y_zero_fp32_mul_yscale = y_zero_fp32 * mWeightScale;

        VARP z_sub_bias;
        if (mWeightZero->getInfo() && mWeightZero->getInfo()->dim.size() > 0) {
            auto sub2 = _MatMul(x_reduce1, _Unsqueeze(y_zero_fp32_mul_yscale, {0}));
            z_sub_bias = convInt8 - sub2;
        } else {
            auto sub2 = x_reduce1 * y_zero_fp32_mul_yscale;
            z_sub_bias = convInt8 - sub2;
        }
        auto z_sub_xzero = sub1 * mDynamicQuantOutputs[1];
        auto z = z_sub_bias * mDynamicQuantOutputs[1] - z_sub_xzero;
        
        auto newExpr = z->expr().first;
        newExpr->setName(expr->name());
        Expr::replace(expr, newExpr);
        return true;
    };

   TemplateMerge::getInstance("Merge").insertTemplate("DynamicQuantMatMulIntegerV2", match, transform, PASS_PRIORITY_HIGH);
    
}
static DynamicQuantMatmulV2 g_dynamic_quant_matmul_v2;

} // namespace Express
} // namespace MNN
