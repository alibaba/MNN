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

bool IsCast(EXPRP expr) {
    const Op* op = expr->get();
    return op && op->type() == OpType_Cast;
}

bool IsConcat(EXPRP expr) {
    const Op* op = expr->get();
    return op && op->type() == OpType_Concat;
}

bool IsReshape(EXPRP expr) {
    const Op* op = expr->get();
    return op && op->type() == OpType_Reshape;
}

bool IsUnsqueeze(EXPRP expr) {
    const Op* op = expr->get();
    return op && op->type() == OpType_Unsqueeze;
}

bool IsTranspose(EXPRP expr) {
    const Op* op = expr->get();
    return op && (op->type() == OpType_Transpose || op->type() == OpType_Permute);
}

bool IsScatterNd(EXPRP expr) {
    const Op* op = expr->get();
    return op && op->type() == OpType_ScatterNd;
}

bool IsMatMul(EXPRP expr) {
    const Op* op = expr->get();
    return op && op->type() == OpType_MatMul;
}

bool IsSoftmax(EXPRP expr) {
    const Op* op = expr->get();
    return op && op->type() == OpType_Softmax;
}

bool IsSelect(EXPRP expr) {
    const Op* op = expr->get();
    return op && op->type() == OpType_Select;
}

bool IsGatherV2(EXPRP expr) {
    const Op* op = expr->get();
    return op && op->type() == OpType_GatherV2;
}

bool IsSlice(EXPRP expr) {
    const Op* op = expr->get();
    return op && (op->type() == OpType_Slice || op->type() == OpType_StridedSlice || op->type() == OpType_SliceTf);
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

bool IsBinaryRealDiv(EXPRP expr) {
    IS_BINARY_OP_TYPE(BinaryOpOperation_REALDIV);
}

bool IsBinarySquaredDifference(Express::EXPRP expr) {
    IS_BINARY_OP_TYPE(BinaryOpOperation_SquaredDifference);
}

bool IsUnarySquare(EXPRP expr) {
    IS_UNARY_OP_TYPE(UnaryOpOperation_SQUARE);
}

bool IsBinaryPow(EXPRP expr) {
    IS_BINARY_OP_TYPE(BinaryOpOperation_POW);
}

bool IsUnarySqrt(EXPRP expr) {
    IS_UNARY_OP_TYPE(UnaryOpOperation_SQRT);
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

bool IsBroadcastTo(EXPRP expr) {
    const Op* op = expr->get();
    return op && op->type() == OpType_BroadcastTo;
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
            if (output.get() == nullptr) {
                continue;
            }
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
