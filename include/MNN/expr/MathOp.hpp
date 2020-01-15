//
//  MathOp.hpp
//  MNN
//
//  Created by MNN on 2019/06/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

namespace MNN {
namespace Express {
//BinaryOPs
MNN_PUBLIC VARP _Add(VARP x, VARP y);
MNN_PUBLIC VARP _Subtract(VARP x, VARP y);    
MNN_PUBLIC VARP _Multiply(VARP x, VARP y);
MNN_PUBLIC VARP _Divide(VARP x, VARP y);
MNN_PUBLIC VARP _Pow(VARP x, VARP y);
MNN_PUBLIC VARP _Minimum(VARP x, VARP y);
MNN_PUBLIC VARP _Maximum(VARP x, VARP y);
MNN_PUBLIC VARP _BiasAdd(VARP value, VARP bias);
MNN_PUBLIC VARP _Greater(VARP x, VARP y);
MNN_PUBLIC VARP _GreaterEqual(VARP x, VARP y);
MNN_PUBLIC VARP _Less(VARP x, VARP y);
MNN_PUBLIC VARP _FloorDiv(VARP x, VARP y);
MNN_PUBLIC VARP _SquaredDifference(VARP x, VARP y);
MNN_PUBLIC VARP _Equal(VARP x, VARP y);
MNN_PUBLIC VARP _LessEqual(VARP x, VARP y);
MNN_PUBLIC VARP _FloorMod(VARP x, VARP y);

//UnaryOPs
MNN_PUBLIC VARP _Sign(VARP a);
MNN_PUBLIC VARP _Abs(VARP x);
MNN_PUBLIC VARP _Negative(VARP x);
MNN_PUBLIC VARP _Floor(VARP x);
MNN_PUBLIC VARP _Round(VARP x);
MNN_PUBLIC VARP _Ceil(VARP x);
MNN_PUBLIC VARP _Square(VARP x);
MNN_PUBLIC VARP _Sqrt(VARP x);
MNN_PUBLIC VARP _Rsqrt(VARP x);
MNN_PUBLIC VARP _Exp(VARP x);
MNN_PUBLIC VARP _Log(VARP x);
MNN_PUBLIC VARP _Sin(VARP x);
MNN_PUBLIC VARP _Cos(VARP x);
MNN_PUBLIC VARP _Tan(VARP x);
MNN_PUBLIC VARP _Asin(VARP x);
MNN_PUBLIC VARP _Acos(VARP x);
MNN_PUBLIC VARP _Atan(VARP x);
MNN_PUBLIC VARP _Reciprocal(VARP x);
MNN_PUBLIC VARP _Log1p(VARP x);
//Only one but not in UnaryOPs 
MNN_PUBLIC VARP _Tanh(VARP x);
MNN_PUBLIC VARP _Sigmoid(VARP x);


//ReduceOPs
MNN_PUBLIC VARP _ReduceSum(VARP input_variable, INTS axis = {}, bool keepDims = false);
MNN_PUBLIC VARP _ReduceMean(VARP input_variable, INTS axis = {}, bool keepDims = false);
MNN_PUBLIC VARP _ReduceMax(VARP input_variable, INTS axis = {}, bool keepDims = false);
MNN_PUBLIC VARP _ReduceMin(VARP input_variable, INTS axis = {}, bool keepDims = false);
MNN_PUBLIC VARP _ReduceProd(VARP input_variable, INTS axis = {}, bool keepDims = false);
MNN_PUBLIC VARP _ReduceAny(VARP input_variable, INTS axis = {}, bool keepDims = false);
MNN_PUBLIC VARP _ReduceAll(VARP input_variable, INTS axis = {}, bool keepDims = false);

//EltwiseOPs
MNN_PUBLIC VARP _Prod(VARP a, VARP b, std::vector<float> coeff);
MNN_PUBLIC VARP _Sum(VARP a, VARP b, std::vector<float> coeff);
MNN_PUBLIC VARP _Max(VARP a, VARP b, std::vector<float> coeff);
MNN_PUBLIC VARP _Sub(VARP a, VARP b, std::vector<float> coeff);

//OtherOPs
template<typename T>
MNN_PUBLIC VARP _Cast(VARP x) {
    return _Cast(x, halide_type_of<T>());
}
MNN_PUBLIC VARP _Cast(VARP x, halide_type_t dtype);
MNN_PUBLIC VARP _MatMul(VARP a, VARP b, bool tranposeA = false, bool tranposeB = false);
MNN_PUBLIC VARP _Normalize(VARP x, int32_t acrossSpatial, int32_t channelShared, float eps, std::vector<float> scale);
MNN_PUBLIC VARP _ArgMax(VARP input, int axis = 0);   
MNN_PUBLIC VARP _BatchMatMul(VARP x, VARP y, bool adj_x = false, bool adj_y = false);
MNN_PUBLIC VARP _UnravelIndex(VARP indices, VARP dims);
MNN_PUBLIC VARP _ScatterNd(VARP indices, VARP updates, VARP shape);
MNN_PUBLIC VARP _OneHot(VARP indices, VARP depth, VARP onValue, VARP offValue, int axis = -1);
MNN_PUBLIC VARP _BroadcastTo(VARP a, VARP shape);
}; // namespace Express
}; // namespace MNN
