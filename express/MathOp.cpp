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
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/MNNDefine.h>
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
static VARP _Eltwise(VARP a, VARP b, EltwiseType type, std::vector<float> coeff) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type               = OpParameter_Eltwise;
    op->type                    = OpType_Eltwise;
    op->main.value              = new EltwiseT;
    op->main.AsEltwise()->type  = type;
    op->main.AsEltwise()->coeff = coeff;
    return (Variable::create(Expr::create(std::move(op), {a, b})));
}
/*Casts a variable to a new type.
Args:
x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8
dtype: The destination type. The list of supported dtypes is the same as x.
Returns:
A variable with same shape as x and same type as dtype.
*/
VARP _Cast(VARP x, halide_type_t dtype) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type                 = OpParameter_CastParam;
    op->type                      = OpType_Cast;
    op->main.value                = new CastParamT;
    op->main.AsCastParam()->dstT  = _convertDataType(dtype);
    return (Variable::create(Expr::create(std::move(op), {x})));
}

/*Computes the absolute value of a variable.
Given a variable of integer or floating-point values, this operation returns a variable of the same type,
where each element contains the absolute value of the corresponding element in the input.
x = MNN.const((-1.0, -2.0, 3.0), (3, ))
x = MNN.abs(x)  # (1.0, 2.0, 3.0)
Args:
x: A variable of type Halide_Type_Int or Halide_Type_Float
Returns:
A variable the same size, type as x with absolute values.
*/
VARP _Abs(VARP x)
{
    return _Unary(x, UnaryOpOperation_ABS);
}
/*Computes numerical negative value element-wise.
x = MNN.const((-1.0, -2.0, 3.0), (3, ))
x = MNN.negative(x) #(1.0, 2.0, -3.0)
Args:
x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Negative(VARP x)
{
    return _Unary(x, UnaryOpOperation_NEG);
}
/*Returns element-wise largest integer not greater than x.
Args:
x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Floor(VARP x)
{
    return _Unary(x, UnaryOpOperation_FLOOR);
}
/*Returns element-wise smallest integer not less than x.
Args:
x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Ceil(VARP x)
{
    return _Unary(x, UnaryOpOperation_CEIL);
}

/*Returns element-wise rounded integer not less than x.
Args:
x: A variable. Must be Halide_Type_Float
Returns:
A variable. Halide_Type_Float.
*/
VARP _Round(VARP x) {
    return _Unary(x, UnaryOpOperation_ROUND);
}

/*Computes square of x element-wise.
Args:
x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Square(VARP x)
{
    return _Unary(x, UnaryOpOperation_SQUARE);
}

/*Computes square root of x element-wise.
Args:
x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Sqrt(VARP x)
{
    return _Unary(x, UnaryOpOperation_SQRT);
}

/*Computes reciprocal of square root of x element-wise.
Args:
x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Rsqrt(VARP x)
{
    return _Unary(x, UnaryOpOperation_RSQRT);
}

/*Computes exponential of x element-wise.
Args:
x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Exp(VARP x)
{
    return _Unary(x, UnaryOpOperation_EXP);
}

/*Computes natural logarithm of x element-wise.
Args:
x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Log(VARP x)
{
    return _Unary(x, UnaryOpOperation_LOG);
}

/*Computes sine of x element-wise.
Given an input variable, this function computes sine of every element in the variable.
Input range is (-inf, inf) and output range is [-1,1].
Args:
x: A variable. Must be one of the following types: Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Sin(VARP x)
{
    return _Unary(x, UnaryOpOperation_SIN);
}

/*Computes cos of x element-wise.
Given an input variable, this function computes cosine of every element in the variable.
Input range is (-inf, inf) and output range is [-1,1]. If input lies outside the boundary, nan is returned.
Args:
x: A variable. Must be one of the following types: Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Cos(VARP x)
{
    return _Unary(x, UnaryOpOperation_COS);
}

/*Computes tan of x element-wise.
Given an input variable, this function computes tangent of every element in the variable.
Input range is (-inf, inf) and output range is (-inf, inf). If input lies outside the boundary, nan is returned.
Args:
x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Tan(VARP x)
{
    return _Unary(x, UnaryOpOperation_TAN);
}

/*Computes the trignometric inverse sine of x element-wise.
The asin operation returns the inverse of sin, such that if y = sin(x) then, x = asin(y).
Note: The output of asin will lie within the invertible range of sine, i.e [-pi/2, pi/2].
Args:
x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Asin(VARP x)
{
    return _Unary(x, UnaryOpOperation_ASIN);
}
/*Computes acos of x element-wise.
Args:
x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
Note: The output of atan will lie within the invertible range of tan, i.e (0.0, pi).
Returns:
A variable. Has the same type as x.
*/
VARP _Acos(VARP x)
{
    return _Unary(x, UnaryOpOperation_ACOS);
}


/*Computes sign of x eltment-wise
 sign(x) = 0 if x=0
 sign(x) =-1 if x<0
 sign(x) = 1 if x>0
 */
VARP _Sign(VARP x) {
    return _Unary(x, UnaryOpOperation_SIGN);
}

/*Computes the trignometric inverse tangent of x element-wise.
The atan operation returns the inverse of tan, such that if y = tan(x) then, x = atan(y).
Note: The output of atan will lie within the invertible range of tan, i.e (-pi/2, pi/2).
Args:
x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Atan(VARP x)
{
    return _Unary(x, UnaryOpOperation_ATAN);
}

/*Computes the reciprocal of x element-wise.
Args:
x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Reciprocal(VARP x)
{
    return _Unary(x, UnaryOpOperation_RECIPROCAL);
}

/*Computes natural logarithm of (1 + x) element-wise.
Args:
x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Log1p(VARP x)
{
    return _Unary(x, UnaryOpOperation_LOG1P);
}

/*Computes hyperbolic tangent of x element-wise.
Given an input variable, this function computes hyperbolic tangent of every element in the variable.
Input range is [-inf, inf] and output range is [-1,1].
Args:
x: A variable. Must be one of the following types: Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Tanh(VARP x) {
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_TanH;
    return (Variable::create(Expr::create(op.get(), {x})));
}
/*Computes sigmoid of x element-wise.
Args:
x: A variable. Must be one of the following types: Halide_Type_Float
Returns:
A variable. Has the same type as x.
*/
VARP _Sigmoid(VARP x) {
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_Sigmoid;
    return (Variable::create(Expr::create(op.get(), {x})));
}


/*Returns x + y element-wise.
Args:
x: A variable. Must be one of the following types:
Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8.
y: A variable. Must have the same type as x.
Returns:
A variable. Has the same type as x.
*/
VARP _Add(VARP x, VARP y) {
  return _Binary(x, y, BinaryOpOperation_ADD);
}

/*Returns x - y element-wise.
Args:
x: A variable. Must be one of the following types:
Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8.
y: A variable. Must have the same type as x.
Returns:
A variable. Has the same type as x.
*/
VARP _Subtract(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_SUB);
}

/*Returns x * y element-wise.
Args:
x: A variable. Must be one of the following types:
Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8.
y: A variable. Must have the same type as x.
Returns:
A variable. Has the same type as x.
*/
VARP _Multiply(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_MUL);
}

/*Computes Python style division of x by y.
Args:
x: A variable. Must be one of the following types:
Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8.
y: A variable. Must have the same type as x.
Returns:
A variable. Has the same type as x.
*/
VARP _Divide(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_REALDIV);
}

/*Computes the power of one value to another.
Args:
x: A variable. Must be one of the following types:
Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64
y: A variable. Must be one of the following types:
Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64
Returns:
A variable. Has the same type as x.
*/
VARP _Pow(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_POW);
}

/*Returns the min of x and y (i.e. x < y ? x : y) element-wise.
Args:
x: A variable. Must be one of the following types:
Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64
y: A variable. Must have the same type as x.
Returns:
A variable. Has the same type as x.
*/
VARP _Minimum(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_MINIMUM);
}
/*Returns the max of x and y (i.e. x > y ? x : y) element-wise.
Args:
x: A variable. Must be one of the following types:
Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64
y: A variable. Must have the same type as x.
Returns:
A variable. Has the same type as x.
*/
VARP _Maximum(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_MAXIMUM);
}

/*Adds bias to value.
This is (mostly) a special case of add where bias is restricted to 1-D.
Broadcasting is supported, so value may have any number of dimensions.
Unlike add, the type of bias is allowed to differ from value in the case where both types are quantized.
Args:
value: A variable with type Halide_Type_Float, Halide_Type_Int
bias: A 1-D variable with size matching the channel dimension of value.
Must be the same type as value unless value is a quantized type, in which case a different quantized type may be used.
Returns:
A variable with the same type as value.
*/
VARP _BiasAdd(VARP value, VARP bias) {
    return _Add(value, bias);
}

/*Returns the truth value of (x > y) element-wise.
Args:
x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
y: A variable. Must have the same type as x.
Returns:
A variable of type bool.
*/

VARP _Greater(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_GREATER);
}

/*Returns the truth value of (x >= y) element-wise.
Args:
x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
y: A variable. Must have the same type as x.
Returns:
A variable of type bool.
*/

VARP _GreaterEqual(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_GREATER_EQUAL);
}

/*Returns the truth value of (x < y) element-wise.
Args:
x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
y: A variable. Must have the same type as x.
Returns:
A variable of type bool.
*/

VARP _Less(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_LESS);
}

/*Returns the value of (x // y) element-wise.
Args:
x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
y: A variable. Must have the same type as x.
Returns:
A variable. Has the same type as x.
*/

VARP _FloorDiv(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_FLOORDIV);
}

/*Returns the value of (x - y)(x - y) element-wise.
Args:
x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
y: A variable. Must have the same type as x.
Returns:
A variable. Has the same type as x.
*/

VARP _SquaredDifference(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_SquaredDifference);
}

/*Returns the truth value of (x == y) element-wise.
Args:
x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
y: A variable. Must have the same type as x.
Returns:
A variable of type bool.
*/

VARP _Equal(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_EQUAL);
}

/*Returns the truth value of (x <= y) element-wise.
Args:
x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
y: A variable. Must have the same type as x.
Returns:
A variable of type bool.
*/

VARP _LessEqual(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_LESS_EQUAL);
}

/*Returns element-wise remainder of division
Args:
x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
y: A variable. Must have the same type as x.
Returns:
A variable. Has the same type as x.
*/

VARP _FloorMod(VARP x, VARP y) {
    return _Binary(x, y, BinaryOpOperation_FLOORMOD);
}

/*Computes the sum of elements across dimensions of a variable
Reduces input_variable along the dimensions given in axis.
Unless keepdims is true, the rank of the variable is reduced by 1 for each entry in axis.
If keepdims is true, the reduced dimensions are retained with length 1.
If axis is empty, all dimensions are reduced, and a variable with a single element is returned.
Args:
input_variable: The variable to reduce. Should have numeric type.
axis: The dimensions to reduce. If empty(the default), reduces all dimensions.
       Must be in the range [-rank(input_variable), rank(input_variable)).
keepdims: If true, retains reduced dimensions with length 1.
Returns:
The reduced variable, of the same dtype as the input_variable.
*/
VARP _ReduceSum(VARP input_variable, INTS axis, bool keepdims) {
    return _Reduce(input_variable, axis, ReductionType_SUM, keepdims);
}
//ruhuan:TODO: ReductionType_ASUM and ReductionType_SUMSQ



/*Computes the mean of elements across dimensions of a variable.
Reduces input_variable along the dimensions given in axis.
Unless keepdims is true, the rank of the variable is reduced by 1 for each entry in axis.
If keepdims is true, the reduced dimensions are retained with length 1.
If axis is empty, all dimensions are reduced, and a variable with a single element is returned.
Args:
input_variable: The variable to reduce. Should have numeric type.
axis: The dimensions to reduce. If empty(the default), reduces all dimensions.
       Must be in the range [-rank(input_variable), rank(input_variable)).
keepdims: If true, retains reduced dimensions with length 1.
Returns:
The reduced variable, of the same dtype as the input_variable.
*/
VARP _ReduceMean(VARP input_variable, INTS axis, bool keepdims) {
    return _Reduce(input_variable, axis, ReductionType_MEAN, keepdims);
}

/*Computes the variance of elements across dimensions of a variable.
Reduces input_variable along the dimensions given in axis.
Unless keepdims is true, the rank of the variable is reduced by 1 for each entry in axis.
If keepdims is true, the reduced dimensions are retained with length 1.
If axis is empty, all dimensions are reduced, and a variable with a single element is returned.
Args:
input_variable: The variable to reduce. Should have numeric type.
axis: The dimensions to reduce. If empty(the default), reduces all dimensions.
       Must be in the range [-rank(input_variable), rank(input_variable)).
keepdims: If true, retains reduced dimensions with length 1.
Returns:
The reduced variable, of the same dtype as the input_variable.
*/
VARP _ReduceVariance(VARP input_variable, INTS axis, bool keepdims) {
    auto mean = _ReduceMean(input_variable, axis, true); // to use broadcast of subtract
    auto variance = _ReduceMean(_Square(_Subtract(input_variable, mean)), axis, keepdims);
    return variance;
}

/*Computes the maximum of elements across dimensions of a variable.
Reduces input_variable along the dimensions given in axis.
Unless keepdims is true, the rank of the variable is reduced by 1 for each entry in axis.
If keepdims is true, the reduced dimensions are retained with length 1.
If axis is empty, all dimensions are reduced, and a variable with a single element is returned.
Args:
input_variable: The variable to reduce. Should have numeric type.
axis: The dimensions to reduce. If empty(the default), reduces all dimensions.
       Must be in the range [-rank(input_variable), rank(input_variable)).
keepdims: If true, retains reduced dimensions with length 1.
Returns:
The reduced variable, of the same dtype as the input_variable.
*/
VARP _ReduceMax(VARP input_variable, INTS axis, bool keepdims) {
    return _Reduce(input_variable, axis, ReductionType_MAXIMUM, keepdims);
}

/*Computes the minimum of elements across dimensions of a variable.
Reduces input_variable along the dimensions given in axis.
Unless keepdims is true, the rank of the variable is reduced by 1 for each entry in axis.
If keepdims is true, the reduced dimensions are retained with length 1.
If axis is empty, all dimensions are reduced, and a variable with a single element is returned.
Args:
input_variable: The variable to reduce. Should have numeric type.
axis: The dimensions to reduce. If empty(the default), reduces all dimensions.
       Must be in the range [-rank(input_variable), rank(input_variable)).
keepdims: If true, retains reduced dimensions with length 1.
Returns:
The reduced variable, of the same dtype as the input_variable.
*/
VARP _ReduceMin(VARP input_variable, INTS axis, bool keepdims) {
    return _Reduce(input_variable, axis, ReductionType_MINIMUM, keepdims);
}

/*Computes the product of elements across dimensions of a variable.
Reduces input_variable along the dimensions given in axis.
Unless keepdims is true, the rank of the variable is reduced by 1 for each entry in axis.
If keepdims is true, the reduced dimensions are retained with length 1.
If axis is empty, all dimensions are reduced, and a variable with a single element is returned.
Args:
input_variable: The variable to reduce. Should have numeric type.
axis: The dimensions to reduce. If empty(the default), reduces all dimensions.
       Must be in the range [-rank(input_variable), rank(input_variable)).
keepdims: If true, retains reduced dimensions with length 1.
Returns:
The reduced variable, of the same dtype as the input_variable.
*/
VARP _ReduceProd(VARP input_variable, INTS axis, bool keepdims) {
    return _Reduce(input_variable, axis, ReductionType_PROD, keepdims);
}
/*Computes the "logical or" of elements across dimensions of a variable.
Reduces input_variable along the dimensions given in axis.
Unless keepdims is true, the rank of the variable is reduced by 1 for each entry in axis.
If keepdims is true, the reduced dimensions are retained with length 1.
If axis is empty, all dimensions are reduced, and a variable with a single element is returned.
Args:
input_variable: The variable to reduce. Should have booling type.
axis: The dimensions to reduce. If empty(the default), reduces all dimensions.
       Must be in the range [-rank(input_variable), rank(input_variable)).
keepdims: If true, retains reduced dimensions with length 1.
Returns:
The reduced variable, of the same dtype as the input_variable.
*/
VARP _ReduceAny(VARP input_variable, INTS axis, bool keepdims) {
    return _Reduce(input_variable, axis, ReductionType_ANY, keepdims);
}
/*Computes the "logical and" of elements across dimensions of a variable.
Reduces input_variable along the dimensions given in axis.
Unless keepdims is true, the rank of the variable is reduced by 1 for each entry in axis.
If keepdims is true, the reduced dimensions are retained with length 1.
If axis is empty, all dimensions are reduced, and a variable with a single element is returned.
Args:
input_variable: The variable to reduce. Should have booling type.
axis: The dimensions to reduce. If empty(the default), reduces all dimensions.
       Must be in the range [-rank(input_variable), rank(input_variable)).
keepdims: If true, retains reduced dimensions with length 1.
Returns:
The reduced variable, of the same dtype as the input_variable.
*/
VARP _ReduceAll(VARP input_variable, INTS axis, bool keepdims) {
    return _Reduce(input_variable, axis, ReductionType_ALL, keepdims);
}

/*Multiply the matrix "a" by the matrix "b".
The inputs must be two-dimensional matrices and the inner dimension of "a" (after being transposed if transpose_a is true)
must match the outer dimension of "b" (after being transposed if transposed_b is true).
Arguments:
a: a variable representing a matrix "a"
b: a variable representing a matrix "b"
tranposeA: If true, "a" is transposed before multiplication.
tranposeB: If true, "b" is transposed before multiplication.
Returns:
The product variable.
*/
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
/* Compute the element-wise prod
Args:
a: A variable. Must be one of the following types: Halide_Type_Float
b: A variable. Must be one of the following types: Halide_Type_Float
coeff: blob-wise coefficients
Returns:
The prod variable.
*/
VARP _Prod(VARP a, VARP b, std::vector<float> coeff) {
    return _Eltwise(a, b, EltwiseType_PROD, coeff);
}
/* Compute the element-wise sum
Args:
a: A variable. Must be one of the following types: Halide_Type_Float
b: A variable. Must be one of the following types: Halide_Type_Float
coeff: blob-wise coefficients
Returns:
The sum variable.
*/
VARP _Sum(VARP a, VARP b, std::vector<float> coeff) {
    return _Eltwise(a, b, EltwiseType_SUM, coeff);
}
/* Compute the element-wise max
Args:
a: A variable. Must be one of the following types: Halide_Type_Float
b: A variable. Must be one of the following types: Halide_Type_Float
coeff: blob-wise coefficients
Returns:
The max variable.
*/
VARP _Max(VARP a, VARP b, std::vector<float> coeff) {
    return _Eltwise(a, b, EltwiseType_MAXIMUM, coeff);
}
/* Compute the element-wise sub
Args:
a: A variable. Must be one of the following types: Halide_Type_Float
b: A variable. Must be one of the following types: Halide_Type_Float
coeff: blob-wise coefficients
Returns:
The sub variable.
*/
VARP _Sub(VARP a, VARP b, std::vector<float> coeff) {
    return _Eltwise(a, b, EltwiseType_SUB, coeff);
}


/*Returns the index with the largest value across axes of a tensor.
Args: input: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
      axis: A int.
            must be in the range -rank(input), rank(input)). Describes which axis of the input variable to reduce across.
            For vectors, use axis = 0.
Returns:
A variable of type int.
*/
VARP _ArgMax(VARP input, int axis) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type                         = OpParameter_ArgMax;
    op->type                              = OpType_ArgMax;
    op->main.value                        = new ArgMaxT;
    op->main.AsArgMax()->axis = axis;
    op->main.AsArgMax()->outMaxVal = 0;
    op->main.AsArgMax()->topK = 0;
    op->main.AsArgMax()->softmaxThreshold = 0;
    return (Variable::create(Expr::create(std::move(op), {input})));

}

/*Multiplies slices of two variable in batches
Multiplies all slices of variable x and y (each slice can be viewed as an element of a batch),
and arranges the individual results in a single output variable of the same batch size.
Each of the individual slices can optionally be adjointed (to adjoint a matrix means to transpose and conjugate it)
before multiplication by setting the adj_x or adj_y flag to True, which are by default False.
The input variable x and y are 2-D or higher with shape [..., r_x, c_x] and [..., r_y, c_y].
The output variable is 2-D or higher with shape [..., r_o, c_o], where:
r_o = c_x if adj_x else r_x
c_o = r_y if adj_y else c_y
It is computed as:
output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
Arguments:
x: 2-D or higher with shape [..., r_x, c_x].
y: 2-D or higher with shape [..., r_y, c_y].
Optional:
adj_x: If True, adjoint the slices of x. Defaults to False.
adj_y: If True, adjoint the slices of y. Defaults to False.
Returns:
Output: 3-D or higher with shape [..., r_o, c_o]
*/
VARP _BatchMatMul(VARP x, VARP y, bool adj_x, bool adj_y) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type                         = OpParameter_BatchMatMulParam;
    op->type                              = OpType_BatchMatMul;
    op->main.value                        = new BatchMatMulParamT;
    op->main.AsBatchMatMulParam()->adjX = adj_x;
    op->main.AsBatchMatMulParam()->adjY = adj_y;

    return (Variable::create(Expr::create(std::move(op), {x, y})));
}


VARP _UnravelIndex(VARP indices, VARP dims) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type  = OpParameter_NONE;
    op->type       = OpType_UnravelIndex;
    op->main.value = nullptr;

    return (Variable::create(Expr::create(std::move(op), {indices, dims})));
}

VARP _ScatterNd(VARP indices, VARP updates, VARP shape) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type  = OpParameter_NONE;
    op->type       = OpType_ScatterNd;
    op->main.value = nullptr;
    return (Variable::create(Expr::create(std::move(op), {indices, updates, shape})));
}

VARP _OneHot(VARP indices, VARP depth, VARP onValue, VARP offValue, int axis) {
    std::unique_ptr<OpT> op(new OpT);
    op->type                       = OpType_OneHot;
    op->main.type                  = OpParameter_OneHotParam;
    op->main.value                 = new OneHotParamT;
    op->main.AsOneHotParam()->axis = axis;

    return (Variable::create(Expr::create(std::move(op), {indices, depth, onValue, offValue})));
}

VARP _BroadcastTo(VARP a, VARP shape) {
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_BroadcastTo;
    op->main.type  = OpParameter_NONE;
    op->main.value = nullptr;
    return (Variable::create(Expr::create(std::move(op), {a, shape})));
}
} // namespace Express
} // namespace MNN
