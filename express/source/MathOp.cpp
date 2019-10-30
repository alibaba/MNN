//
//  MathOp.cpp
//  MNN
//
//  Created by MNN on 2019/06/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <map>
#include <numeric>
#include "ExprCreator.hpp"
#include "MNNDefine.h"
#include "MNN_generated.h"

namespace MNN {
namespace Express {
static DataType _convertDataType(halide_type_t type) {
    if (type.code == halide_type_float) {
        return DataType_DT_FLOAT;
    }
    if (type.code == halide_type_uint && type.bits == 8) {
        return DataType_DT_UINT8;
    }
    if (type.code == halide_type_int && type.bits == 8) {
        return DataType_DT_INT8;
    }
    if (type.code == halide_type_int && type.bits == 32) {
        return DataType_DT_INT32;
    }
    return DataType_DT_INVALID;
}
VARP _Cast(VARP a, halide_type_t srcType, halide_type_t dstType) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type                 = OpParameter_CastParam;
    op->type                      = OpType_Cast;
    op->main.value                = new CastParamT;
    op->main.AsCastParam()->srcT  = _convertDataType(srcType);
    op->main.AsCastParam()->dstT  = _convertDataType(dstType);
    return (Variable::create(Expr::create(std::move(op), {a})));
}

static VARP _Binary(VARP x, VARP y, BinaryOpOperation operation) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type                 = OpParameter_BinaryOp;
    op->type                      = OpType_BinaryOp;
    op->main.value                = new BinaryOpT;
    op->main.AsBinaryOp()->opType = operation;
    op->main.AsBinaryOp()->T      = DataType_DT_FLOAT;
    return (Variable::create(Expr::create(op.get(), {x, y})));
}
static VARP _Unary(VARP x, UnaryOpOperation operation) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type                = OpParameter_UnaryOp;
    op->type                     = OpType_UnaryOp;
    op->main.value               = new UnaryOpT;
    op->main.AsUnaryOp()->opType = operation;
    op->main.AsUnaryOp()->T      = DataType_DT_FLOAT;
    return (Variable::create(Expr::create(op.get(), {x})));
}
VARP _Mul(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_MUL);
}
VARP _Div(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_REALDIV);
}
VARP _Sub(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_SUB);
}
VARP _Add(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_ADD);
}
VARP _Neg(VARP x) {
    return _Unary(x, UnaryOpOperation_NEG);
}
VARP _Rsqrt(VARP x) {
    return _Unary(x, UnaryOpOperation_RSQRT);
}
VARP _Log(VARP x) {
    return _Unary(x, UnaryOpOperation_LOG);
}
VARP _Exp(VARP x) {
    return _Unary(x, UnaryOpOperation_EXP);
}
VARP _Square(VARP x) {
    return _Unary(x, UnaryOpOperation_SQUARE);
}

VARP _Tanh(VARP x) {
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_TanH;
    return (Variable::create(Expr::create(op.get(), {x})));
}

VARP _Sigmoid(VARP x) {
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_Sigmoid;
    return (Variable::create(Expr::create(op.get(), {x})));
}
static VARP _Reduce(VARP x, INTS dim, ReductionType type, bool keepDim) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type                         = OpParameter_ReductionParam;
    op->type                              = OpType_Reduction;
    op->main.value                        = new ReductionParamT;
    op->main.AsReductionParam()->dType    = DataType_DT_FLOAT;
    op->main.AsReductionParam()->operation= type;
    op->main.AsReductionParam()->dim      = dim;
    op->main.AsReductionParam()->keepDims = keepDim;
    return (Variable::create(Expr::create(op.get(), {x})));
}
VARP _ReduceMax(VARP x, INTS dim, bool keepDim) {
    return _Reduce(x, dim, ReductionType_MAXIMUM, keepDim);
}
VARP _ReduceMin(VARP x, INTS dim, bool keepDim) {
    return _Reduce(x, dim, ReductionType_MINIMUM, keepDim);
}
VARP _Sum(VARP x, INTS dim, bool keepDim) {
    return _Reduce(x, dim, ReductionType_SUM, keepDim);
}
VARP _Mean(VARP x, INTS dim, bool keepDim) {
    return _Reduce(x, dim, ReductionType_MEAN, keepDim);
}
VARP _Prod(VARP x, INTS dim, bool keepDim) {
    return _Reduce(x, dim, ReductionType_PROD, keepDim);
}
VARP _Any(VARP x, INTS dim, bool keepDim) {
    return _Reduce(x, dim, ReductionType_ANY, keepDim);
}
VARP _All(VARP x, INTS dim, bool keepDim) {
    return _Reduce(x, dim, ReductionType_ALL, keepDim);
}
VARP _MatMul(VARP a, VARP b, bool tranposeA, bool tranposeB) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type                   = OpParameter_MatMul;
    op->type                        = OpType_MatMul;
    op->main.value                  = new MatMulT;
    op->main.AsMatMul()->transposeA = tranposeA;
    op->main.AsMatMul()->transposeB = tranposeB;
    return (Variable::create(Expr::create(op.get(), {a, b})));
}
    
VARP _Normalize(VARP x, int32_t acrossSpatial, int32_t channelShared, float eps, std::vector<float> scale) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type                         = OpParameter_Normalize;
    op->type                              = OpType_Normalize;
    op->main.value                        = new NormalizeT;
    op->main.AsNormalize()->acrossSpatial = acrossSpatial;
    op->main.AsNormalize()->channelShared = channelShared;
    op->main.AsNormalize()->eps           = eps;
    op->main.AsNormalize()->scale         = scale;
    return (Variable::create(Expr::create(std::move(op), {x})));
}
    
static VARP _Eltwise(VARP a, VARP b, EltwiseType type, std::vector<float> coeff) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type               = OpParameter_Eltwise;
    op->type                    = OpType_Eltwise;
    op->main.value              = new EltwiseT;
    op->main.AsEltwise()->type  = type;
    op->main.AsEltwise()->coeff = coeff;
    return (Variable::create(Expr::create(std::move(op), {a, b})));
}
    
VARP _Prod(VARP a, VARP b, std::vector<float> coeff) {
    return _Eltwise(a, b, EltwiseType_PROD, coeff);
}
VARP _Sum(VARP a, VARP b, std::vector<float> coeff) {
    return _Eltwise(a, b, EltwiseType_SUM, coeff);
}
VARP _Max(VARP a, VARP b, std::vector<float> coeff) {
    return _Eltwise(a, b, EltwiseType_MAXIMUM, coeff);
}
VARP _Sub(VARP a, VARP b, std::vector<float> coeff) {
    return _Eltwise(a, b, EltwiseType_SUB, coeff);
}
    
} // namespace Express
} // namespace MNN
