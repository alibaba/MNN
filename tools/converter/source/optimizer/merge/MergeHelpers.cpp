//
//  MergeHelpers.cpp
//  MNNConverter
//
//  Created by MNN on b'2020/07/20'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <unordered_map>
#include <vector>

#include "../../common/Common.hpp"
#include "MNN_generated.h"
#include "MergeHelpers.hpp"

using namespace MNN::Express;

namespace MNN {
namespace helpers {

bool IsConstant(EXPRP expr) {
    const Op* op = expr->get();
    if ((op && op->type() == OpType_Const) || (!op && expr->inputType() == VARP::CONSTANT)) {
        return true;
    }
    return false;
}

bool IsBinaryOp(EXPRP expr) {
    const Op* op = expr->get();
    return op && op->type() == OpType_BinaryOp;
}

bool IsUnaryOp(EXPRP expr) {
    const Op* op = expr->get();
    return op && op->type() == OpType_UnaryOp;
}

#define IS_BINARY_OP_TYPE(op_type)                        \
    if (!IsBinaryOp(expr)) {                              \
        return false;                                     \
    }                                                     \
    int type = expr->get()->main_as_BinaryOp()->opType(); \
    return type == op_type;

#define IS_UNARY_OP_TYPE(op_type)                        \
    if (!IsUnaryOp(expr)) {                              \
        return false;                                    \
    }                                                    \
    int type = expr->get()->main_as_UnaryOp()->opType(); \
    return type == op_type;

bool IsBinaryAdd(EXPRP expr) {
    IS_BINARY_OP_TYPE(BinaryOpOperation_ADD);
}

bool IsBinarySub(EXPRP expr) {
    IS_BINARY_OP_TYPE(BinaryOpOperation_SUB);
}

bool IsBinaryMul(EXPRP expr) {
    IS_BINARY_OP_TYPE(BinaryOpOperation_MUL);
}

bool IsBinarySquaredDifference(Express::EXPRP expr) {
    IS_BINARY_OP_TYPE(BinaryOpOperation_SquaredDifference);
}

bool IsUnarySquare(EXPRP expr) {
    IS_UNARY_OP_TYPE(UnaryOpOperation_SQUARE);
}

bool IsUnaryRsqrt(EXPRP expr) {
    IS_UNARY_OP_TYPE(UnaryOpOperation_RSQRT);
}

bool IsUnaryNeg(EXPRP expr) {
    IS_UNARY_OP_TYPE(UnaryOpOperation_NEG);
}

#undef IS_BINARY_OP_TYPE
#undef IS_UNARY_OP_TYPE

bool IsReductionMean(EXPRP expr) {
    const Op* op = expr->get();
    if (!op || op->type() != OpType_Reduction) {
        return false;
    }
    int type = op->main_as_ReductionParam()->operation();
    return type == ReductionType_MEAN;
}

bool IsConvolution(EXPRP expr) {
    const Op* op = expr->get();
    return op && op->type() == OpType_Convolution;
}

bool IsExpandDims(EXPRP expr) {
    const Op* op = expr->get();
    return op && op->type() == OpType_ExpandDims;
}

EXPRP InputExpr(EXPRP expr, int input_index) {
    return expr->inputs().at(input_index)->expr().first;
}

EXPRP OutputExpr(EXPRP expr, int output_index) {
    return expr->outputs().at(output_index).lock();
}

std::vector<VARP> OutputVars(EXPRP expr) {
    std::unordered_map<int, VARP> outputs;
    for (WeakEXPRP w : expr->outputs()) {
        EXPRP child = w.lock();
        if (!child.get()) {
            continue;
        }
        for (VARP output : child->inputs()) {
            int output_index = 0;
            EXPRP parent;
            std::tie(parent, output_index) = output->expr();
            if (parent.get() == expr.get()) {
                outputs.emplace(output_index, output);
            }
        }
    }
    std::vector<VARP> v_outputs;
    for (const auto& it : outputs) {
        int index = 0;
        VARP output;
        std::tie(index, output) = it;
        if (!output.get()) {
            continue;
        }
        if (v_outputs.size() <= index) {
            v_outputs.resize(index + 1);
        }
        v_outputs[index] = output;
    }
    return std::move(v_outputs);
}

VARP ConvertLayout(VARP input, Dimensionformat dest_layout, Dimensionformat src_layout) {
    std::unique_ptr<OpT> convert(new OpT);
    convert->type                               = OpType_ConvertTensor;
    convert->main.type                          = OpParameter_TensorConvertInfo;
    convert->main.value                         = new TensorConvertInfoT;
    convert->main.AsTensorConvertInfo()->dest   = convertFormat(dest_layout);
    convert->main.AsTensorConvertInfo()->source = convertFormat(src_layout);
    return (Variable::create(Expr::create(convert.get(), {input})));
}

} // namespace helpers
} // namespace MNN
