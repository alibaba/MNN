# MathOp
```cpp
class MathOp
```
## 成员函数

---
### _Add
```cpp
MNN_PUBLIC VARP _Add(VARP x, VARP y);
```
返回x + y的值

参数：
- `x` 一个变量，Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8类型之一
- `y` 一个变量，Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8类型之一

返回：x相同的类型变量

---
### _Subtract
```cpp
MNN_PUBLIC VARP _Subtract(VARP x, VARP y);
```
计算x-y的值

参数：
- `x` 一个变量，Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8类型之一
- `y` 一个变量，Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8类型之一

返回：x相同的类型变量

---
### _Multiply
```cpp
MNN_PUBLIC VARP _Multiply(VARP x, VARP y);
```
计算x*y的值

参数：
- `x` 一个变量，Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8类型之一
- `y` 一个变量，Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8类型之一

返回：x相同的类型变量

---
### _Divide
```cpp
MNN_PUBLIC VARP _Divide(VARP x, VARP y);
```
计算x/y的值

参数：
- `x` 一个变量，Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8类型之一
- `y` 一个变量，Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8类型之一

返回：x相同的类型变量

---
### _Pow
```cpp
MNN_PUBLIC VARP _Pow(VARP x, VARP y);
```
计算x的y的幂次方的值

参数：
- `x` 一个变量，Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64类型之一
- `y` 一个变量，Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64类型之一

返回：x相同的类型变量

---
### _Minimum
```cpp
MNN_PUBLIC VARP _Minimum(VARP x, VARP y);
```
返回x和y的最小值

参数：
- `x` 一个变量，Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64类型之一
- `y` 一个变量，Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64类型之一

返回：x相同的类型变量

---
### _Maximum
```cpp
MNN_PUBLIC VARP _Maximum(VARP x, VARP y);
```
返回x和y的最大值

参数：
- `x` 一个变量，Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64类型之一
- `y` 一个变量，Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64类型之一

返回：x相同的类型变量

---
### _BiasAdd
```cpp
MNN_PUBLIC VARP _BiasAdd(VARP value, VARP bias);
```
增加value的偏差。这(主要)是加法的一个特殊情况，其中偏差限制在1-D。支持广播，因此value可以有任意数量的维度。与加法不同，在量子化的情况下，偏差的类型允许与值不同

参数：
- `value` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `bias` 一个一维变量，其大小与值的通道维度相匹配，必须与值的类型相同，除非值是量化类型，在这种情况下可以使用不同的量化类型

返回：value相同的类型变量

---
### _Greater
```cpp
MNN_PUBLIC VARP _Greater(VARP x, VARP y);
```
比较x和y的大小，如果x > y为true，否者为false

参数：
- `x` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `y` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一

返回：true或者false

---
### _GreaterEqual
```cpp
MNN_PUBLIC VARP _GreaterEqual(VARP x, VARP y);
```
比较x和y的大小，如果x >= y为true，否者为false

参数：
- `x` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `y` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一

返回：true或者false

---
### _Less
```cpp
MNN_PUBLIC VARP _Less(VARP x, VARP y);
```
比较x和y的大小，如果x < y为true，否者为false

参数：
- `x` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `y` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一

返回：true或者false

---
### _FloorDiv
```cpp
MNN_PUBLIC VARP _FloorDiv(VARP x, VARP y);
```
返回x // y的值


参数：
- `x` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `y` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一

返回：x相同的类型变量

---
### _SquaredDifference
```cpp
MNN_PUBLIC VARP _SquaredDifference(VARP x, VARP y);
```
返回(x - y)(x - y)的值

参数：
- `x` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `y` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一

返回：x相同的类型变量

---
### _Equal
```cpp
MNN_PUBLIC VARP _Equal(VARP x, VARP y);
```
判断x和y是否相等，如果x = y为true，否者为false


参数：
- `x` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `y` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一

返回：true或者false

---
### _LessEqual
```cpp
MNN_PUBLIC VARP _LessEqual(VARP x, VARP y);
```
判断x和y的大小，如果x <= y则返回true，否则返回false

参数：
- `x` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `y` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一

返回：true或者false

---
### _FloorMod
```cpp
MNN_PUBLIC VARP _FloorMod(VARP x, VARP y);
```
返回x % y的值


参数：
- `x` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `y` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一

返回：x相同的类型变量

---
### _Atan2
```cpp
MNN_PUBLIC VARP _Atan2(VARP x, VARP y);
```
计算y / x的元素反正切的值

参数：
- `x` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `y` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一

返回：x相同的类型变量

---
### _LogicalOr
```cpp
MNN_PUBLIC VARP _LogicalOr(VARP x, VARP y);
```
返回x和y的逻辑或的值

参数：
- `x` Halide_Type_Int类型的变量
- `y` Halide_Type_Int类型的变量

返回：true/false

---
### _NotEqual
```cpp
MNN_PUBLIC VARP _NotEqual(VARP x, VARP y);
```
判断x和y是否相等，如果x != y则返回true，否者返回false

参数：
- `x` Halide_Type_Int类型的变量
- `y` Halide_Type_Int类型的变量

返回：true/false

---
### _BitwiseAnd
```cpp
MNN_PUBLIC VARP _BitwiseAnd(VARP x, VARP y);
```
返回x和y的按位逻辑与的值(x & y)


参数：
- `x` Halide_Type_Int类型的变量
- `y` Halide_Type_Int类型的变量

返回：x相同的类型变量

---
### _BitwiseOr
```cpp
MNN_PUBLIC VARP _BitwiseOr(VARP x, VARP y);
```
返回x和y的按位逻辑或的值(x | y)

参数：
- `x` Halide_Type_Int类型的变量
- `y` Halide_Type_Int类型的变量

返回：x相同的类型变量

---
### _BitwiseXor
```cpp
MNN_PUBLIC VARP _BitwiseXor(VARP x, VARP y);
```
返回x和y的按位异或的值(x ^ y)


参数：
- `x` Halide_Type_Int类型的变量
- `y` Halide_Type_Int类型的变量

返回：x相同的类型变量

---
### _Sign
```cpp
MNN_PUBLIC VARP _Sign(VARP a);
```
去掉x元素的符号
sign(x) = 0 if x=0
sign(x) =-1 if x<0
sign(x) = 1 if x>0

参数：
- `a` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：-1、0或者1

---
### _Abs
```cpp
MNN_PUBLIC VARP _Abs(VARP x);
```
计算变量的绝对值，给定一个整型或浮点型变量，该操作将返回一个相同类型的变量，其中每个元素都包含输入中对应元素的绝对值
x = MNN.const((-1.0， -2.0, 3.0)， (3，))
x = MNN.abs(x) # (1.0, 2.0, 3.0)


参数：
- `x` Halide_Type_Int或Halide_Type_Float类型的变量

返回：一个大小相同的变量，类型与x的绝对值相同

---
### _Negative
```cpp
MNN_PUBLIC VARP _Negative(VARP x);
```
计算元素数值负值
x = MNN.const((-1.0， -2.0, 3.0)， (3，))
x = MNN.negative(x) #(1.0, 2.0， -3.0)

参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：和x相同类型的变量

---
### _Floor
```cpp
MNN_PUBLIC VARP _Floor(VARP x);
```
返回不大于x的最大整数

参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：和x相同类型的变量

---
### _Round
```cpp
MNN_PUBLIC VARP _Round(VARP x);
```
返回元素四舍五入的整数

参数：
- `x` Halide_Type_Float类型的变量

返回：Halide_Type_Float类型的变量

---
### _Ceil
```cpp
MNN_PUBLIC VARP _Ceil(VARP x);
```
返回不小于x的最小整数

参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：和x相同类型的变量

---
### _Square
```cpp
MNN_PUBLIC VARP _Square(VARP x);
```
计算x元素的平方值

参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：x相同类型的变量

---
### _Sqrt
```cpp
MNN_PUBLIC VARP _Sqrt(VARP x);
```
计算x的平方根

参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：x相同类型的变量

---
### _Rsqrt
```cpp
MNN_PUBLIC VARP _Rsqrt(VARP x);
```
计算x根号的倒数

参数：一个变量，Halide_Type_Int或Halide_Type_Float类型之一
- `x` 

返回：x相同类型的变量

---
### _Exp
```cpp
MNN_PUBLIC VARP _Exp(VARP x);
```
计算x的指数

参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：x相同类型的变量

---
### _Log
```cpp
MNN_PUBLIC VARP _Log(VARP x);
```
计算x的对数

参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：x相同类型的变量

---
### _Sin
```cpp
MNN_PUBLIC VARP _Sin(VARP x);
```
计算x的正弦值

参数：
- `x` Halide_Type_Float类型的变量

返回：x相同类型的变量

---
### _Sinh
```cpp
MNN_PUBLIC VARP _Sinh(VARP x);
```
计算x的双曲正弦值

参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：x相同类型的变量

---
### _Cos
```cpp
MNN_PUBLIC VARP _Cos(VARP x);
```
计算x的余弦值

参数：
- `x` Halide_Type_Float类型的变量

返回：x相同类型的变量

---
### _Cosh
```cpp
MNN_PUBLIC VARP _Cosh(VARP x);
```
计算x的双曲余弦的值


参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：x相同类型的变量

---
### _Tan
```cpp
MNN_PUBLIC VARP _Tan(VARP x);
```
计算x的正切值

参数：
- `x` Halide_Type_Float类型的变量

返回：x相同类型的变量

---
### _Asin
```cpp
MNN_PUBLIC VARP _Asin(VARP x);
```
计算x的反正弦值

参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：x相同类型的变量

---
### _Asinh
```cpp
MNN_PUBLIC VARP _Asinh(VARP x);
```
计算x的反双曲正弦的值

参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：x相同类型的变量

---
### _Acos
```cpp
MNN_PUBLIC VARP _Acos(VARP x);
```
计算x的反余弦值

参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：x相同类型的变量

---
### _Acosh
```cpp
MNN_PUBLIC VARP _Acosh(VARP x);
```
计算x的反双曲余弦的值

参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：x相同类型的变量

---
### _Atan
```cpp
MNN_PUBLIC VARP _Atan(VARP x);
```
计算x的反正切函数

参数：
- `x` Halide_Type_Float类型的变量

返回：x相同类型的变量

---
### _Atanh
```cpp
MNN_PUBLIC VARP _Atanh(VARP x);
```
计算x的双曲反正切的值

参数：
- `x` Halide_Type_Float类型的变量

返回：x相同类型的变量

---
### _Reciprocal
```cpp
MNN_PUBLIC VARP _Reciprocal(VARP x);
```
计算x的倒数

参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：x相同类型的变量

---
### _Log1p
```cpp
MNN_PUBLIC VARP _Log1p(VARP x);
```
计算(1 + x)的自然对数


参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：x相同类型的变量

---
### _Gelu
```cpp
MNN_PUBLIC VARP _Gelu(VARP x);
```
计算x的高斯误差线性单元激活函数


参数：
- `x` Halide_Type_Float类型的变量

返回：x相同类型的变量

---
### _Tanh
```cpp
MNN_PUBLIC VARP _Tanh(VARP x);
```
计算x的双曲正切函数

参数：
- `x` Halide_Type_Float类型的变量

返回：x相同类型的变量

---
### _Sigmoid
```cpp
MNN_PUBLIC VARP _Sigmoid(VARP x);
```
计算x的神经元的非线性作用函数

参数：
- `x` Halide_Type_Float类型的变量

返回：x相同类型的变量

---
### _Erf
```cpp
MNN_PUBLIC VARP _Erf(VARP x);
```
计算x的高斯误差值

参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：x相同类型的变量

---
### _Erfc
```cpp
MNN_PUBLIC VARP _Erfc(VARP x);
```
计算x的互补误差函数的值

参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：x相同类型的变量

---
### _Erfinv
```cpp
MNN_PUBLIC VARP _Erfinv(VARP x);
```
x的逆函数的值

参数：
- `x` 一个变量，Halide_Type_Int或Halide_Type_Float类型之一

返回：x相同类型的变量

---
### _Expm1
```cpp
MNN_PUBLIC VARP _Expm1(VARP x);
```
计算((x指数)- 1)的值

参数：
- `x` Halide_Type_Float类型的变量

返回：x相同类型的变量

---
### _Hardswish
```cpp
MNN_PUBLIC VARP _Hardswish(VARP x);
```
元素x的Hardswish神经网络激活函数

参数：
- `x` Halide_Type_Float类型的变量

返回：x相同类型的变量

---
### _ReduceSum
```cpp
MNN_PUBLIC VARP _ReduceSum(VARP input_variable, INTS axis = {}, bool keepDims = false);
```
计算变量各维度上元素的和，沿轴中给定的维度减少input_variable。除非keepdim为真，否则变量在轴上的每一项的排名都会减少1。如果keepdim为true，则缩减后的维度保留长度为1；如果axis为空，则减少所有维度，并返回具有单个元素的变量

参数：
- `input_variable` 要减少的变量，应该是数值类型
- `axis` 要减少的尺寸。如果为空(默认值)，则减少所有维度。必须在范围内[-rank(input_variable)， rank(input_variable))
- `keepDims` 如果为true，则保留长度为1的缩减维度

返回：简化后的变量，与input_variable具有相同的类型

---
### _ReduceMean
```cpp
MNN_PUBLIC VARP _ReduceMean(VARP input_variable, INTS axis = {}, bool keepDims = false);
```
计算变量各维度元素的平均值，沿轴中给定的维度减少input_variable，除非keepdim为真，否则变量在轴上的每一项的排名都会减少1。如果keepdim为true，则缩减后的维度保留长度为1；如果axis为空，则减少所有维度，并返回具有单个元素的变量

参数：
- `input_variable` 要减少的变量，应该是数值类型
- `axis` 要减少的尺寸。如果为空(默认值)，则减少所有维度。必须在范围内[-rank(input_variable)， rank(input_variable))
- `keepDims` 如果为true，则保留长度为1的缩减维度

返回：简化后的变量，与input_variable具有相同的类型

---
### _ReduceMax
```cpp
MNN_PUBLIC VARP _ReduceMax(VARP input_variable, INTS axis = {}, bool keepDims = false);
```
计算变量跨维元素的最大值，沿轴中给定的维度减少input_variable，除非keepdim为真，否则变量在轴上的每一项的排名都会减少1。如果keepdim为true，则缩减后的维度保留长度为1；如果axis为空，则减少所有维度，并返回具有单个元素的变量

参数：
- `input_variable` 要减少的变量，应该是数值类型
- `axis` 要减少的尺寸。如果为空(默认值)，则减少所有维度。必须在范围内[-rank(input_variable)， rank(input_variable))
- `keepDims` 如果为true，则保留长度为1的缩减维度

返回：简化后的变量，与input_variable具有相同的类型

---
### _ReduceMin
```cpp
MNN_PUBLIC VARP _ReduceMin(VARP input_variable, INTS axis = {}, bool keepDims = false);
```
计算变量跨维元素的最小值，沿轴中给定的维度减少input_variable，除非keepdim为真，否则变量在轴上的每一项的排名都会减少1。如果keepdim为true，则缩减后的维度保留长度为1；如果axis为空，则减少所有维度，并返回具有单个元素的变量

参数：
- `input_variable` 要减少的变量，应该是数值类型
- `axis` 要减少的尺寸。如果为空(默认值)，则减少所有维度。必须在范围内[-rank(input_variable)， rank(input_variable))
- `keepDims` 如果为true，则保留长度为1的缩减维度

返回：简化后的变量，与input_variable具有相同的类型

---
### _ReduceProd
```cpp
MNN_PUBLIC VARP _ReduceProd(VARP input_variable, INTS axis = {}, bool keepDims = false);
```
计算变量各维度上元素的乘积，沿轴中给定的维度减少input_variable，除非keepdim为真，否则变量在轴上的每一项的排名都会减少1。如果keepdim为true，则缩减后的维度保留长度为1；如果axis为空，则减少所有维度，并返回具有单个元素的变量

参数：
- `input_variable` 要减少的变量，应该是数值类型
- `axis` 要减少的尺寸。如果为空(默认值)，则减少所有维度。必须在范围内[-rank(input_variable)， rank(input_variable))
- `keepDims` 如果为true，则保留长度为1的缩减维度

返回：简化后的变量，与input_variable具有相同的类型

---
### _ReduceAny
```cpp
MNN_PUBLIC VARP _ReduceAny(VARP input_variable, INTS axis = {}, bool keepDims = false);
```
跨变量的维度计算元素的“逻辑或”的值，沿轴中给定的维度减少input_variable，除非keepdim为真，否则变量在轴上的每一项的排名都会减少1。如果keepdim为true，则缩减后的维度保留长度为1；如果axis为空，则减少所有维度，并返回具有单个元素的变量

参数：
- `input_variable` 要减少的变量，应该是数值类型
- `axis` 要减少的尺寸。如果为空(默认值)，则减少所有维度。必须在范围内[-rank(input_variable)， rank(input_variable))
- `keepDims` 如果为true，则保留长度为1的缩减维度

返回：简化后的变量，与input_variable具有相同的类型

---
### _ReduceAll
```cpp
MNN_PUBLIC VARP _ReduceAll(VARP input_variable, INTS axis = {}, bool keepDims = false);
```
跨变量的维度计算元素的“逻辑和”的值，沿轴中给定的维度减少input_variable，除非keepdim为真，否则变量在轴上的每一项的排名都会减少1。如果keepdim为true，则缩减后的维度保留长度为1；如果axis为空，则减少所有维度，并返回具有单个元素的变量

参数：
- `input_variable` 要减少的变量，应该是数值类型
- `axis` 要减少的尺寸。如果为空(默认值)，则减少所有维度。必须在范围内[-rank(input_variable)， rank(input_variable))
- `keepDims` 如果为true，则保留长度为1的缩减维度

返回：简化后的变量，与input_variable具有相同的类型

---
### _ReduceSumMutable
```cpp
MNN_PUBLIC VARP _ReduceSumMutable(VARP input_variable, INTS axis = {}, bool keepDims = false);
```
计算变量各维度上元素的和，沿轴中给定的维度减少input_variable。除非keepdim为真，否则变量在轴上的每一项的排名都会减少1。如果keepdim为true，则缩减后的维度保留长度为1；如果axis为空，则减少所有维度，并返回具有单个元素的变量，是可变的

参数：
- `input_variable` 要减少的变量，应该是数值类型
- `axis` 要减少的尺寸。如果为空(默认值)，则减少所有维度。必须在范围内[-rank(input_variable)， rank(input_variable))
- `keepDims` 如果为true，则保留长度为1的缩减维度

返回：简化后的变量，与input_variable具有相同的类型

---
### _ReduceMeanMutable
```cpp
MNN_PUBLIC VARP _ReduceMeanMutable(VARP input_variable, INTS axis = {}, bool keepDims = false);
```
计算变量各维度元素的平均值，沿轴中给定的维度减少input_variable，除非keepdim为真，否则变量在轴上的每一项的排名都会减少1。如果keepdim为true，则缩减后的维度保留长度为1；如果axis为空，则减少所有维度，并返回具有单个元素的变量

参数：
- `input_variable` 要减少的变量，应该是数值类型
- `axis` 要减少的尺寸。如果为空(默认值)，则减少所有维度。必须在范围内[-rank(input_variable)， rank(input_variable))
- `keepDims` 如果为true，则保留长度为1的缩减维度

返回：简化后的变量，与input_variable具有相同的类型

---
### _ReduceMaxMutable
```cpp
MNN_PUBLIC VARP _ReduceMaxMutable(VARP input_variable, INTS axis = {}, bool keepDims = false);
```
计算变量跨维元素的最大值，沿轴中给定的维度减少input_variable，除非keepdim为真，否则变量在轴上的每一项的排名都会减少1。如果keepdim为true，则缩减后的维度保留长度为1；如果axis为空，则减少所有维度，并返回具有单个元素的变量

参数：
- `input_variable` 要减少的变量，应该是数值类型
- `axis` 要减少的尺寸。如果为空(默认值)，则减少所有维度。必须在范围内[-rank(input_variable)， rank(input_variable))
- `keepDims` 如果为true，则保留长度为1的缩减维度

返回：简化后的变量，与input_variable具有相同的类型

---
### _ReduceMinMutable
```cpp
MNN_PUBLIC VARP _ReduceMinMutable(VARP input_variable, INTS axis = {}, bool keepDims = false);
```
计算变量跨维元素的最大值，沿轴中给定的维度减少input_variable，除非keepdim为真，否则变量在轴上的每一项的排名都会减少1。如果keepdim为true，则缩减后的维度保留长度为1；如果axis为空，则减少所有维度，并返回具有单个元素的变量

参数：
- `input_variable` 要减少的变量，应该是数值类型
- `axis` 要减少的尺寸。如果为空(默认值)，则减少所有维度。必须在范围内[-rank(input_variable)， rank(input_variable))
- `keepDims` 如果为true，则保留长度为1的缩减维度

返回：简化后的变量，与input_variable具有相同的类型

---
### _ReduceProdMutable
```cpp
MNN_PUBLIC VARP _ReduceProdMutable(VARP input_variable, INTS axis = {}, bool keepDims = false);
```
计算变量各维度上元素的乘积，沿轴中给定的维度减少input_variable，除非keepdim为真，否则变量在轴上的每一项的排名都会减少1。如果keepdim为true，则缩减后的维度保留长度为1；如果axis为空，则减少所有维度，并返回具有单个元素的变量

参数：
- `input_variable` 要减少的变量，应该是数值类型
- `axis` 要减少的尺寸。如果为空(默认值)，则减少所有维度。必须在范围内[-rank(input_variable)， rank(input_variable))
- `keepDims` 如果为true，则保留长度为1的缩减维度

返回：简化后的变量，与input_variable具有相同的类型

---
### _ReduceAnyMutable
```cpp
MNN_PUBLIC VARP _ReduceAnyMutable(VARP input_variable, INTS axis = {}, bool keepDims = false);
```
跨变量的维度计算元素的“逻辑或”的值，沿轴中给定的维度减少input_variable，除非keepdim为真，否则变量在轴上的每一项的排名都会减少1。如果keepdim为true，则缩减后的维度保留长度为1；如果axis为空，则减少所有维度，并返回具有单个元素的变量

参数：
- `input_variable` 要减少的变量，应该是数值类型
- `axis` 要减少的尺寸。如果为空(默认值)，则减少所有维度。必须在范围内[-rank(input_variable)， rank(input_variable))
- `keepDims` 如果为true，则保留长度为1的缩减维度

返回：简化后的变量，与input_variable具有相同的类型

---
### _ReduceAllMutable
```cpp
MNN_PUBLIC VARP _ReduceAllMutable(VARP input_variable, INTS axis = {}, bool keepDims = false);
```
跨变量的维度计算元素的“逻辑和”的值，沿轴中给定的维度减少input_variable，除非keepdim为真，否则变量在轴上的每一项的排名都会减少1。如果keepdim为true，则缩减后的维度保留长度为1；如果axis为空，则减少所有维度，并返回具有单个元素的变量

参数：
- `input_variable` 要减少的变量，应该是数值类型
- `axis` 要减少的尺寸。如果为空(默认值)，则减少所有维度。必须在范围内[-rank(input_variable)， rank(input_variable))
- `keepDims` 如果为true，则保留长度为1的缩减维度

返回：简化后的变量，与input_variable具有相同的类型

---
### _Prod
```cpp
MNN_PUBLIC VARP _Prod(VARP a, VARP b, std::vector<float> coeff);
```
计算元素积

参数：
- `a` Halide_Type_Float类型的变量
- `b` Halide_Type_Float类型的变量
- `coeff` blob-wise系数

返回：元素积变量

---
### _Sum
```cpp
MNN_PUBLIC VARP _Sum(VARP a, VARP b, std::vector<float> coeff);
```
计算元素和

参数：
- `a` Halide_Type_Float类型的变量
- `b` Halide_Type_Float类型的变量
- `coeff` blob-wise系数

返回：元素和

---
### _Max
```cpp
MNN_PUBLIC VARP _Max(VARP a, VARP b, std::vector<float> coeff);
```
计算元素最大值

参数：
- `a` Halide_Type_Float类型的变量
- `b` Halide_Type_Float类型的变量
- `coeff` blob-wise系数

返回：最大值

---
### _Sub
```cpp
MNN_PUBLIC VARP _Sub(VARP a, VARP b, std::vector<float> coeff);
```
计算元素下标

参数：
- `a` Halide_Type_Float类型的变量
- `b` Halide_Type_Float类型的变量
- `coeff` blob-wise系数

返回：下标元素

---
### _EltwiseProdInt8
```cpp
MNN_PUBLIC VARP _EltwiseProdInt8(VARP x, VARP y, 
                    std::vector<int8_t> x_weight, std::vector<int32_t> x_bias, std::vector<float> x_scale, std::vector<float> x_tensorScale,
                    std::vector<int8_t> y_weight, std::vector<int32_t> y_bias, std::vector<float> y_scale, std::vector<float> y_tensorScale,
                    std::vector<int8_t> output_weight, std::vector<int32_t> output_bias, std::vector<float> output_scale, std::vector<float> output_tensorScale);
```
在Eltwise层对x和y进行累计乘积


参数：
- `x` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `y` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `x_weight` 变量x的权值
- `x_bias`  变量x的偏差
- `x_scale` 变量x的比例因子
- `x_tensorScale` 变量x的张量比例因子
- `y_weight` 变量y的权值
- `y_bias` 变量y的偏差
- `y_scale` 变量y的比例因子
- `y_tensorScale` 变量y的张量比例因子
- `output_weight` 输出数据的权值
- `output_bias` 输出数据的偏差
- `output_scale` 输出数据的比例因子
- `output_tensorScale` 输出数据的张量比例因子

返回：VARP类型变量

---
### _EltwiseSumInt8
```cpp
MNN_PUBLIC VARP _EltwiseSumInt8(VARP x, VARP y, 
                    std::vector<int8_t> x_weight, std::vector<int32_t> x_bias, std::vector<float> x_scale, std::vector<float> x_tensorScale,
                    std::vector<int8_t> y_weight, std::vector<int32_t> y_bias, std::vector<float> y_scale, std::vector<float> y_tensorScale,
                    std::vector<int8_t> output_weight, std::vector<int32_t> output_bias, std::vector<float> output_scale, std::vector<float> output_tensorScale);
```
在Eltwise层对x和y进行累计求和


参数：
- `x` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `y` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `x_weight` 变量x的权值
- `x_bias`  变量x的偏差
- `x_scale` 变量x的比例因子
- `x_tensorScale` 变量x的张量比例因子
- `y_weight` 变量y的权值
- `y_bias` 变量y的偏差
- `y_scale` 变量y的比例因子
- `y_tensorScale` 变量y的张量比例因子
- `output_weight` 输出数据的权值
- `output_bias` 输出数据的偏差
- `output_scale` 输出数据的比例因子
- `output_tensorScale` 输出数据的张量比例因子

返回：VARP类型变量

---
### _EltwiseSubInt8
```cpp
MNN_PUBLIC VARP _EltwiseSubInt8(VARP x, VARP y, 
                    std::vector<int8_t> x_weight, std::vector<int32_t> x_bias, std::vector<float> x_scale, std::vector<float> x_tensorScale,
                    std::vector<int8_t> y_weight, std::vector<int32_t> y_bias, std::vector<float> y_scale, std::vector<float> y_tensorScale,
                    std::vector<int8_t> output_weight, std::vector<int32_t> output_bias, std::vector<float> output_scale, std::vector<float> output_tensorScale);
```
在Eltwise层对x和y进行累计求差值


参数：
- `x` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `y` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `x_weight` 变量x的权值
- `x_bias`  变量x的偏差
- `x_scale` 变量x的比例因子
- `x_tensorScale` 变量x的张量比例因子
- `y_weight` 变量y的权值
- `y_bias` 变量y的偏差
- `y_scale` 变量y的比例因子
- `y_tensorScale` 变量y的张量比例因子
- `output_weight` 输出数据的权值
- `output_bias` 输出数据的偏差
- `output_scale` 输出数据的比例因子
- `output_tensorScale` 输出数据的张量比例因子

返回：VARP类型变量

---
### _EltwiseMaxInt8
```cpp
MNN_PUBLIC VARP _EltwiseMaxInt8(VARP x, VARP y, 
                    std::vector<int8_t> x_weight, std::vector<int32_t> x_bias, std::vector<float> x_scale, std::vector<float> x_tensorScale,
                    std::vector<int8_t> y_weight, std::vector<int32_t> y_bias, std::vector<float> y_scale, std::vector<float> y_tensorScale,
                    std::vector<int8_t> output_weight, std::vector<int32_t> output_bias, std::vector<float> output_scale, std::vector<float> output_tensorScale);
```
在Eltwise层对x和y进行累计求最大值


参数：
- `x` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `y` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `x_weight` 变量x的权值
- `x_bias`  变量x的偏差
- `x_scale` 变量x的比例因子
- `x_tensorScale` 变量x的张量比例因子
- `y_weight` 变量y的权值
- `y_bias` 变量y的偏差
- `y_scale` 变量y的比例因子
- `y_tensorScale` 变量y的张量比例因子
- `output_weight` 输出数据的权值
- `output_bias` 输出数据的偏差
- `output_scale` 输出数据的比例因子
- `output_tensorScale` 输出数据的张量比例因子

返回：VARP类型变量

---
### _Mod
```cpp
MNN_PUBLIC VARP _Mod(VARP x, VARP y);
```
求余函数，即x和y作除法运算后的余数

参数：
- `x` 一个变量，Halide_Type_Int, Halide_Type_Float类型之一
- `y` 一个变量，Halide_Type_Int, Halide_Type_Float类型之一

返回：和x类型相同的变量

---
### _Cast
```cpp
VARP _Cast(VARP x) {
    return _Cast(x, halide_type_of<T>());
};
```
将变量强制转换为新类型

参数：
- `x` 一个变量，Halide_Type_Int, Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8类型之一

返回：与x形状相同，与dtype类型相同的变量

---
### _Cast
```cpp
MNN_PUBLIC VARP _Cast(VARP x, halide_type_t dtype);
```
将变量强制转换为新类型

参数：
- `x` 一个变量，Halide_Type_Int, Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8类型之一
- `dtype` 目标类型，支持的dtypes列表与x相同

返回：与x形状相同，与dtype类型相同的变量

---
### _MatMul
```cpp
MNN_PUBLIC VARP _MatMul(VARP a, VARP b, bool tranposeA = false, bool tranposeB = false);
```
矩阵a * 矩阵b，输入必须是二维矩阵和“a”的内部维数(如果转置se_a为真，则转置后)，必须匹配"b"的外部尺寸(如果transposed_b为true则被转置)

参数：
- `a` 一个表示矩阵A的变量
- `b` 一个表示矩阵B的变量
- `tranposeA` 如果为true，则a在乘法之前被转置，默认为false
- `tranposeB` 如果为true，则b在乘法之前被转置，默认为false

返回：矩阵a * 矩阵b

---
### _Normalize
```cpp
MNN_PUBLIC VARP _Normalize(VARP x, int32_t acrossSpatial, int32_t channelShared, float eps, std::vector<float> scale);
```
返回x数据转换成指定的标准化的格式

参数：
- `x` 输入变量
- `acrossSpatial` 输入变量
- `channelShared` 输入变量
- `eps` 输入变量，data_format
- `scale` 缩放因子

返回：x数据转换成指定的标准化的格式

---
### _ArgMax
```cpp
MNN_PUBLIC VARP _ArgMax(VARP input, int axis = 0);
```
返回张量坐标轴上最大值的索引

参数：
- `input` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `axis` 坐标轴

返回：索引值

---
### _ArgMin
```cpp
MNN_PUBLIC VARP _ArgMin(VARP input, int axis = 0);
```
返回张量坐标轴上最小值的索引

参数：
- `input` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `axis` 坐标轴

返回：索引值

---
### _BatchMatMul
```cpp
MNN_PUBLIC VARP _BatchMatMul(VARP x, VARP y, bool adj_x = false, bool adj_y = false);
```
批量相乘两个变量的切片，乘以变量x和y的所有切片(每个切片可以看作是一个批处理的一个元素)，并将单个结果安排在同一个批处理大小的单个输出变量中。每一个单独的切片都可以有选择地被伴随(伴随一个矩阵意味着转置和共轭它)将adj_x或adj_y标志设置为True，默认为False。输入变量x和y是二维或更高的形状[…]， r_x, c_x]和[…]、r_y提出)。输出变量为二维或更高的形状[…]， r_o, c_o]，其中:
R_o = c_x if adj_x else r_x
C_o = r_y if adj_y else c_y
计算公式为:
输出[…，:，:] =矩阵(x[…，:，:]) *矩阵(y[…]、::])

参数：
- `x` 二维或更高形状[..., r_x, c_x]
- `y` 二维或更高形状[..., r_x, c_x]
- `adj_x` 如果为True，则连接x的切片，默认为False
- `adj_y` 如果为True，则连接y的切片，默认为False

返回：3-D或更高形状[…], r_o, c_o]

---
### _UnravelIndex
```cpp
MNN_PUBLIC VARP _UnravelIndex(VARP indices, VARP dims);
```
返回indices中的元素在维度为dims的数组中的索引值，默认按元组的形式返回

参数：
- `indices` 指定的张量保存指向输出张量的索引
- `dims` 操作的维度

返回：索引

---
### _ScatterNd
```cpp
MNN_PUBLIC VARP _ScatterNd(VARP indices, VARP updates, VARP shape);
```
根据声明的索引，通过对声明的形状张量的零张量内的单个切片或值进行分散更新，来形成不同的张量

参数：
- `indices` 指定的张量保存指向输出张量的索引
- `updates` 它是声明的张量，用于保存索引的值
- `shape` 它是输出张量的规定形状

返回：张量

---
### _ScatterNd
```cpp
MNN_PUBLIC VARP _ScatterNd(VARP indices, VARP updates, VARP shape, VARP input);
```
根据声明的索引，通过对声明的形状张量的零张量内的单个切片或值进行分散更新，来形成不同的张量

参数：
- `indices` 指定的张量保存指向输出张量的索引
- `updates` 它是声明的张量，用于保存索引的值
- `shape` 它是输出张量的规定形状
- `input` 输入的张量数据

返回：张量

---
### _ScatterNd
```cpp
MNN_PUBLIC VARP _ScatterNd(VARP indices, VARP updates, VARP shape, int reduction);
```
根据声明的索引，通过对声明的形状张量的零张量内的单个切片或值进行分散更新，来形成不同的张量

参数：
- `indices` 指定的张量保存指向输出张量的索引
- `updates` 它是声明的张量，用于保存索引的值
- `shape` 它是输出张量的规定形状
- `reduction` 减少数值，默认为-1

返回：张量

---
### _ScatterNd
```cpp
MNN_PUBLIC VARP _ScatterNd(VARP indices, VARP updates, VARP shape, VARP input, int reduction);
```
根据声明的索引，通过对声明的形状张量的零张量内的单个切片或值进行分散更新，来形成不同的张量

参数：
- `indices` 指定的张量保存指向输出张量的索引
- `updates` 它是声明的张量，用于保存索引的值
- `shape` 它是输出张量的规定形状
- `input` 输入的张量数据
- `reduction` 减少数值，默认为-1

返回：张量

---
### _ScatterElements
```cpp
MNN_PUBLIC VARP _ScatterElements(VARP data, VARP indices, VARP updates, int reduction = -1);
```
根据updates和indices来更新data的值，并把结果返回

参数：
- `data` 一个张量
- `indices` 一个张量
- `updates` 一个张量
- `reduction` 减少数值，默认为-1

返回：更新后的data

---
### _ScatterElements
```cpp
MNN_PUBLIC VARP _ScatterElements(VARP data, VARP indices, VARP updates, VARP axis, int reduction = -1);
```
根据updates和indices来更新data的值，并把结果返回

参数：
- `data` 一个张量
- `indices` 一个张量
- `updates` 一个张量
- `axis` 数轴，表示在行还是列进行操作
- `reduction` 减少数值，默认为-1

返回：更新后的data

---
### _OneHot
```cpp
MNN_PUBLIC VARP _OneHot(VARP indices, VARP depth, VARP onValue, VARP offValue, int axis = -1);
```
独热编码，一般是在有监督学习中对数据集进行标注时候使用的，指的是在分类问题中，将存在数据类别的那一类用X表示，不存在的用Y表示，这里的X常常是1，Y常常是0

参数：
- `indices` 输入的张量
- `depth` 一个标量，用于定位维度的深度
- `onValue` 定义在indices[j] = i 时填充输出值的标量
- `offValue` 定义在indices[j] != i 时填充输出值的标量
- `axis` 要填充的轴，默认为-1

返回：编码数据

---
### _BroadcastTo
```cpp
MNN_PUBLIC VARP _BroadcastTo(VARP a, VARP shape);
```
利用广播将原始矩阵成倍增加

参数：
- `a` 广播的张量
- `shape` 期望输出的形状

返回：矩阵

---
### _LinSpace
```cpp
MNN_PUBLIC VARP _LinSpace(VARP start, VARP stop, VARP num);
```
创建一个等差数列

参数：
- `start` 数据的起始点，即区间的最小值
- `stop` 数据的结束点，即区间的最大值
- `num` 数据量，可以理解成分割了多少份

返回：等差数列

---
### _RandomUnifom
```cpp
MNN_PUBLIC VARP _RandomUnifom(VARP shape, halide_type_t dtype, float low = 0.0f, float high = 1.0f, int seed0 = 0, int seed1 = 0);
```
获取随机数

参数：
- `shape` 输入数据的形状
- `dtype` 目标类型
- `low` 随机数的最小区间值
- `high` 随机数的最大区间值
- `seed0` 随机因子
- `seed1` 随机因子

返回：dtype类型的随机数

---
### _CumSum
```cpp
MNN_PUBLIC VARP _CumSum(VARP x, int axis, bool exclusive = false, bool reverse = false);
```
计算元素x在axis坐标轴的累加值

参数：
- `x` 输入参数
- `axis` 坐标轴
- `exclusive` 默认为false
- `reverse` 是否逆向，默认为false

返回：和x相同类型的变量

---
### _CumProd
```cpp
MNN_PUBLIC VARP _CumProd(VARP x, int axis);
```
计算x在axis坐标轴的累计乘积

参数：
- `x` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一
- `axis` 坐标轴

返回：和x相同类型的变量

---
### _Svd
```cpp
MNN_PUBLIC VARPS _Svd(VARP x);
```
奇异值分解，降维算法中的特征分解

参数：
- `x` 一个变量，Halide_Type_Float, Halide_Type_Int类型之一

返回：和x相同类型的变量

---
### _Histogram
```cpp
MNN_PUBLIC VARP _Histogram(VARP x, int bin, int min, int max, int channel = -1);
```
直方图函数

参数：
- `x` 待统计的数据
- `bin` 指定统计的区间个数
- `min` 统计范围最小值
- `max` 统计范围最大值
- `channel`通道，默认为-1

返回：和x相同类型的变量


