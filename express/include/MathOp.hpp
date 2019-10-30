//
//  MathOp.hpp
//  MNN
//
//  Created by MNN on 2019/06/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

namespace MNN {
namespace Express {
MNN_EXPRESS_PUBLIC VARP _Cast(VARP a, halide_type_t srcType, halide_type_t dstType);
    
MNN_EXPRESS_PUBLIC VARP _Mul(VARP x, VARP y);
MNN_EXPRESS_PUBLIC VARP _Sub(VARP x, VARP y);
MNN_EXPRESS_PUBLIC VARP _Add(VARP x, VARP y);
MNN_EXPRESS_PUBLIC VARP _Div(VARP x, VARP y);
MNN_EXPRESS_PUBLIC VARP _Log(VARP x);
MNN_EXPRESS_PUBLIC VARP _Neg(VARP x);
MNN_EXPRESS_PUBLIC VARP _Rsqrt(VARP x);
MNN_EXPRESS_PUBLIC VARP _Tanh(VARP x);
MNN_EXPRESS_PUBLIC VARP _Exp(VARP x);
MNN_EXPRESS_PUBLIC VARP _Square(VARP x);
MNN_EXPRESS_PUBLIC VARP _Sigmoid(VARP x);

MNN_EXPRESS_PUBLIC VARP _ReduceMin(VARP x, INTS dim, bool keepDim = false);
MNN_EXPRESS_PUBLIC VARP _ReduceMax(VARP x, INTS dim, bool keepDim = false);
MNN_EXPRESS_PUBLIC VARP _Sum(VARP x, INTS dim, bool keepDim = false);
MNN_EXPRESS_PUBLIC VARP _Mean(VARP x, INTS dim, bool keepDim = false);
MNN_EXPRESS_PUBLIC VARP _Prod(VARP x, INTS dim, bool keepDim = false);
MNN_EXPRESS_PUBLIC VARP _Any(VARP x, INTS dim, bool keepDim = false);
MNN_EXPRESS_PUBLIC VARP _All(VARP x, INTS dim, bool keepDim = false);
MNN_EXPRESS_PUBLIC VARP _MatMul(VARP a, VARP b, bool tranposeA = false, bool tranposeB = false);
MNN_EXPRESS_PUBLIC VARP _Normalize(VARP x, int32_t acrossSpatial, int32_t channelShared, float eps, std::vector<float> scale);
    
MNN_EXPRESS_PUBLIC VARP _Prod(VARP a, VARP b, std::vector<float> coeff);
MNN_EXPRESS_PUBLIC VARP _Sum(VARP a, VARP b, std::vector<float> coeff);
MNN_EXPRESS_PUBLIC VARP _Max(VARP a, VARP b, std::vector<float> coeff);
MNN_EXPRESS_PUBLIC VARP _Sub(VARP a, VARP b, std::vector<float> coeff);
    
}; // namespace Express
}; // namespace MNN
