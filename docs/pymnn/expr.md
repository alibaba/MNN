## expr

```python
module expr
```
expr是MNN的表达式模块，包含了一系列的表达式函数能够构造MNN中所有的算子，并提供了类型[Var](Var.md)用来描述数据与表达式。

通过表达式函数和Var变量的组合，可以实现以下功能：
- 模型推理
- 数值计算
- 模型构造

---
### `expr Types`
- [Var](Var.md)
- `var_like` 指`Var`或者可以转换为`Var`的数据，如：`list`，`tuple`，`scalar`等

---
### `const(value_list, shape, data_format, dtype)`
根据输入数据创建一个`Const`类型的`Var`；该函数是创建的`Var`的最基本函数，
能够将`list`，`tuple`，`bytes`，`ndarray`，`PyCapsule`，`int指针`等格式的数据转换成`Var`

*注意：`value_list`仅在PYMNN_NUMPY_USABLE打开的情况下支持`ndarray`，移动端默认关闭*

参数：
- `value_list:ndarray/list/tuple/bytes/PyCapsule/int_addr` 输入数据
- `shape:[int]` 构造`Var`的形状
- `data_format:data_format` 数据排布格式，参考[data_format](Var.html#data-format)
- `dtype:dtype` 数据类型，参考[dtype](Var.html#dtype)

返回：创建的`Var`

返回类型：`Var`

示例：

```python
>>> import numpy as np
>>> expr.const(np.arange(4.0).astype(np.float32), [1, 4], expr.NCHW, expr.float) # ndarray
array([[0., 1., 2., 3.]], dtype=float32)
>>> expr.const([2, 3, 4], [3], expr.NCHW, expr.int) # list/tuple
array([2, 3, 4], dtype=int32)
>>> expr.const(bytes('abc', encoding='utf8'), [3], expr.NCHW, expr.uint8) # bytes
array([97, 98, 99], dtype=uint8)
>>> expr.const(MNN.Tensor([2, 3]).getData(), [2], expr.NCHW, expr.int) # PyCapsule
array([2, 3], dtype=int32)
>>> expr.const(np.arange(4.0).astype(np.float32).__array_interface__['data'][0], [4], expr.NCHW, expr.float) # int_addr 该方法要求ndarray内存必须连续
array([0., 1., 2., 3.], dtype=float32)
```
---
### `set_thread_number(numberThread)`
设置表达式求值的线程数

参数：
- `numberThread:int` 线程数，当`numberThread < 1`时，设置`numberThread=1`，当`numberThread > 8`时，设置`numberThread=8`

返回：`None`

返回类型：`None`

示例：

```python
>>> expr.set_thread_number(4)
```
---
### `load_as_list(fileName)`
从文件中加载模型，并将模型转换为计算图，以`list`的形式返回计算图的所有`Var`节点

参数：
- `fileName:str` 模型文件路径

返回：加载的模型计算图

返回类型：`[Var]`

示例：

```python
>>> len(expr.load_as_list('mobilenet_v1.mnn'))
31
```
---
### `save(vars, fileName, |forInference)`
将`list`形式的计算图存储为模型文件，此函数也用于将`Var`保存到磁盘中

参数：
- `vars:list` 计算图的`list`形式
- `fileName:str` 模型文件路径
- `forInference:bool` 是否仅保存为推理使用，默认为`True`

返回：存储计算图到模型文件中

返回类型：`None`

示例：

```python
>>> x = expr.const([1., 2., 3., 4.], [2, 2])
>>> expr.save([x], 'x.mnn')
>>> expr.load_as_list('x.mnn')
[array([[1., 2.],
        [3., 4.]], dtype=float32)]
```
---
### `gc(full)`
手动回收内存，当在循环中调用MNN表达式求值时，常量部分数据不会在每次循环结束释放，当执行次数增加时会有内存增长现象，可以在每次循环结束时调用该函数回收常量内存

参数：
- `full:bool` 是否全部回收，*目前回收方式`True`和`False`没有区别*

返回：`None`

返回类型：`None`

示例：

```python
>>> expr.gc(1)
```
---
### `lazy_eval(lazy)`
是否开启惰性求值，在Python中默认关闭；主要区别如下：

| 模式 | 开启惰性求值 | 关闭惰性求值 |
|:----|:-----------|:-----------|
| 构建区别 | 创建的`Var`时实际表达式类型 | 创建的`Var`都是`Const`类型 |
| 计算方式 | 对`Var`执行read操作时触发计算 | 构建表达式时会立即执行计算 |
| 应用场景 | 使用表达式构建模型，训练，需要对表达式进行修改 | 数值计算，关闭后会加速和降低内存占用 |

参数：
- `lazy:bool` 是否开启惰性求值

返回：`None`

返回类型：`None`

示例：

```python
>>> expr.placeholder().op_type
'Const'
>>> expr.lazy_eval(True)
>>> expr.placeholder().op_type
'Input'
```
---
### `set_lazy_mode(mode)`
设置惰性计算的模式，仅在开启惰性求值的状态下生效，

- 0 : 所有计算均延迟执行
- 1 : 立即进行几何计算，内容计算延迟执行，适用于构建静态模型或训练时求导

默认为0


参数：
- `x:int` 模式类型

返回：`None`

返回类型：`None`

示例：
```python
>>> expr.lazy_eval(True)
>>> expr.set_lazy_mode(0)
>>> y = expr.concat([x], -1)
>>> expr.save([y], "concat.mnn") # 模型中为 concat 算子
>>> expr.set_lazy_mode(1)
>>> y = expr.concat([x], -1)
>>> expr.save([y], "concat_static.mnn") # 模型中为 raster 算子
```

---
### `set_global_executor_config(backend, precision, threadnum)`
设置expr运行后端、精度、线程数(gpu代表mode)：

参数：
- `backend:int` 例如：0->CPU 1->Metal 2->CUDA 3->OPENCL 
- `precision:int` 例如：0—>Normal 1->High 2->Low 
- `threadnum:int` 例如：CPU表示线程数  GPU表示Mode

返回：`None`

返回类型：`None`

示例：

```python
>>> expr.set_global_executor_config(2, 2, 1)
```

---
### `sync()`
MNN VARP同步，调用后可以保证改VARP计算完毕

返回：`None`

返回类型：`None`

示例：

```python
>>> mnn_var = expr.placeholder([2,2])
>>> mnn_var.sync()
```

---
### `set_device_ptr(device_ptr, memory_type)`
设置MNN VARP GPU内存地址，同时指定给定内存地址对应的内存类型(CUDA/OPENCL/OPENGL等)，仅在MNN VARP有GPU内存时可用：

参数：
- `device_ptr:uint64_t` 整形内存指针地址
- `memory_type:int` 例如： 2->CUDA 3->OpenCL等, 详见include/MNN/MNNForwardType.h文件中MNNForwardType结构体

返回：`None`

返回类型：`None`

示例：

```python
>>> torch_tensor = torch.empty([1, 1000], dtype=torch.float16).cuda()
>>> mnn_var = expr.placeholder([2,2])
>>> mnn_var.set_device_ptr(torch_tensor.data_ptr() ,2)
```

---
### `copy_to_device_ptr(device_ptr, memory_type)`
拷贝MNN VARP GPU内存到指定内存地址, 同时指定给定内存地址对应的内存类型(CUDA/OPENCL/OPENGL等)：

参数：
- `device_ptr:uint64_t` 整形内存指针地址
- `memory_type:int` 例如： 2->CUDA 3->OpenCL等, 详见include/MNN/MNNForwardType.h文件中MNNForwardType结构体

返回：`None`

返回类型：`None`

示例：

```python
>>> torch_tensor = torch.empty([1, 1000], dtype=torch.float16).cuda()
>>> mnn_var = expr.placeholder([2,2])
>>> mnn_var.copy_to_device_ptr(torch_tensor.data_ptr() ,2)
```

---
### `sign(x)`
返回输入值的符号，正数返回1，负数返回-1

参数：
- `x:Var_like` 输入变量

返回：x的符号，`1`或`-1`

返回类型：`Var`

示例：

```python
>>> expr.sign([-5., 4.5])
    array([-1., 1.])
```

---
### `abs(x)`
返回输入值的绝对值，正数返回原值，负数返回相反数

参数：
- `x:Var_like` 输入变量

返回：x的绝对值

返回类型：`Var`

示例：

```python
>>> expr.abs([-5., 4.5])
    array([5., 4.5])
```

---
### `negative(x)`
返回输入值的相反数

参数：
- `x:Var_like` 输入变量

返回：x的相反数

返回类型：`Var`

示例：

```python
>>> expr.abs([-5., 4.5])
    array([5., -4.5])
```

---
### `floor(x)`
返回不大于输入值的最大整数

参数：
- `x:Var_like` 输入变量

返回：不大于x的最大整数

返回类型：`Var`

示例：

```python
>>> expr.floor([-5.1, 4.5])
array([-6.,  4.])
```

---
### `round(x)`
返回输入值的四舍五入的值

参数：
- `x:Var_like` 输入变量

返回：x的四舍五入的值

返回类型：`Var`

示例：

```python
>>> expr.round([-5.1, 4.5])
array([-5.,  5.])
```

---
### `ceil(x)`
返回输入值的整数部分的值

参数：
- `x:Var_like` 输入变量

返回：x的整数部分的值

返回类型：`Var`

示例：

```python
>>> expr.ceil([-4.9, 4.5])
array([-4.,  5.])
```

---
### `square(x)`
返回输入值的平方值

参数：
- `x:Var_like` 输入变量

返回：x的平方值

返回类型：`Var`

示例：

```python
 >>> expr.square([-5., 4.5])
array([25., 20.25])
```

---
### `sqrt(x)`
返回输入值的平方根的值，输入值为非负实数

参数：
- `x:Var_like` 输入变量

返回：x的平方根的值

返回类型：`Var`

示例：

```python
>>> expr.sqrt([9., 4.5])
array([3., 2.1213202])
```

---
### `rsqrt(x)`
返回输入值的平方根的倒数的值，输入值为非负实数

参数：
- `x:Var_like` 输入变量

返回：x的平方根的倒数的值

返回类型：`Var`

示例：

```python
>>> expr.rsqrt([9., 4.5])
array([0.33333334, 0.47140455])
```

---
### `exp(x)`
返回自然常数e的输入值(x)次方的值

参数：
- `x:Var_like` 输入变量

返回：自然常数e的x次方的值

返回类型：`Var`

示例：

```python
>>> expr.exp([9., 4.5])
array([8102.449, 90.01698])
```

---
### `log(x)`
返回输入值以自然常数e为底的对数值

参数：
- `x:Var_like` 输入变量

返回：x的以自然常数e为底的对数值

返回类型：`Var`

示例：

```python
>>> expr.log([9., 4.5])
array([2.1972246, 1.5040774])
```

---
### `sin(x)`
返回输入值的正弦值

参数：
- `x:Var_like` 输入变量

返回：x的正弦值

返回类型：`Var`

示例：

```python
>>> expr.sin([9., 4.5])
array([0.4121185, -0.9775301])
```

---
### `sinh(x)`
返回输入值的双曲正弦值
相当于 1/2 * (np.exp(x) - np.exp(-x)) 或者 -1j * np.sin(1j*x)

参数：
- `x:Var_like` 输入变量

返回：x的双曲正弦值

返回类型：`Var`

示例：

```python
>>> expr.sinh([9., 4.5])
array([4051.542, 45.00301])
```

---
### `cos(x)`
返回输入值的余弦值

参数：
- `x:Var_like` 输入变量

返回：x的余弦值

返回类型：`Var`

示例：

```python
>>> expr.cos([9., 4.5])
array([-0.91113025, -0.2107958])
```

---
### `cosh(x)`
返回输入值的双曲余弦值
相当于 1/2 * (np.exp(x) + np.exp(-x)) 或者 np.cos(1j*x)

参数：
- `x:Var_like` 输入变量

返回：x的双曲余弦值

返回类型：`Var`

示例：

```python
>>> expr.cosh([9., 4.5])
array([4051.542, 45.014122])
```

---
### `tan(x)`
返回输入值的正切值

参数：
- `x:Var_like` 输入变量

返回：x的正切值

返回类型：`Var`

示例：

```python
>>> expr.tan([9., 4.5])
array([-0.45231566, 4.637332])
```

---
### `tanh(x)`
返回输入值的双曲正切值
相当于 np.sinh(x)/np.cosh(x) 或者 -1j * np.tan(1j*x)

参数：
- `x:Var_like` 输入变量

返回：x的双曲正切值

返回类型：`Var`

示例：

```python
>>> expr.tanh([9., 4.5])
array([1., 0.9997533])
```

---
### `asin(x)`
返回输入值的反正弦值，别名arcsin

参数：
- `x:Var_like` 输入变量

返回：x的反正弦值

返回类型：`Var`

示例：

```python
>>> expr.asin([9., 0.5])
array([nan, 0.5235988])
```

---
### `asinh(x)`
返回输入值的双曲反正弦值，别名arcsinh

参数：
- `x:Var_like` 输入变量

返回：x的双曲反正弦值

返回类型：`Var`

示例：

```python
>>> expr.asinh([9., 0.5])
array([2.893444, 0.4812118])
```

---
### `acos(x)`
返回输入值的反余弦值，别名arccos

参数：
- `x:Var_like` 输入变量

返回：x的反余弦值

返回类型：`Var`

示例：

```python
>>> expr.asin([9., 0.5])
array([nan, 1.0471975])
```

---
### `acosh(x)`
返回输入值的双曲反余弦值，别名arccosh

参数：
- `x:Var_like` 输入变量

返回：x的双曲反余弦值

返回类型：`Var`

示例：

```python
>>> expr.acosh([9., 0.5])
array([2.887271, nan])
```

---
### `atan(x)`
返回输入值的反正切值，别名arctan

参数：
- `x:Var_like` 输入变量

返回：x的反正切值

返回类型：`Var`

示例：

```python
>>> expr.atan([9., 0.5])
array([1.4601392, 0.4636476])
```

---
### `atanh(x)`
返回输入值的双曲反正切值，别名arctanh

参数：
- `x:Var_like` 输入变量

返回：x的双曲反正切值

返回类型：`Var`

示例：

```python
>>> expr.atanh([9., 0.5])
array([1.4601392, 0.4636476])
```

---
### `reciprocal(x)`
返回输入值的倒数，输入值不能为0

参数：
- `x:Var_like` 输入变量

返回：x的倒数

返回类型：`Var`

示例：

```python
>>> expr.reciprocal([9., 0.5])
array([0.11111111, 2.])
```

---
### `log1p(x)`
返回log(1 + x)的值

参数：
- `x:Var_like` 输入变量

返回：log(1 + x)的值

返回类型：`Var`

示例：

```python
>>> expr.log1p([9., 0.5])
array([2.3025851, 0.4054651])
```

### `gelu(x)`
返回 0.5x(1+tanh(sqrt(pi/2)*(0.044715*x^3))) 的值

参数：
- `x:Var_like` 输入变量

返回：0.5x(1+tanh(sqrt(pi/2)*(0.044715*x^3))) 的值

返回类型：`Var`

示例：

```python
>>> expr.gelu([9., 0.5])
array([9., 0.345714])
```

---
### `sigmoid(x)`
返回 1/(1+exp(-1) 的值

参数：
- `x:Var_like` 输入变量

返回：1/(1+exp(-1)的值

返回类型：`Var`

示例：

```python
>>> expr.sigmoid([9., 0.5])
array([0.9998766, 0.62246716])
```

---
### `erf(x)`
计算高斯误差函数值的数据输入

参数：
- `x:Var_like` 输入变量

返回：高斯误差函数值的数据

返回类型：`Var`

示例：

```python
>>> expr.erf([[1., 25., 16., 0.]])
array([0.8427007., 1., 1., 0.])
```

---
### `erfc(x)`
返回 x 的互补误差

参数：
- `x:Var_like` 输入变量

返回： x的互补误差

返回类型：`Var`

示例：

```python
>>> expr.erfc([0.67., 1.34., -6.])
array([0.34337229769969496., 0.05808628474163466., 2.0.])
```

---
### `erfinv(x)`
返回x的逆误差

参数：
- `x:Var_like` 输入变量

返回：x的逆误差

返回类型：`Var`

示例：

```python
>>> expr.erfinv([0.1., 0.2])
array([0.08885599., 0.17914345.])
```

---
### `expm1(x)`
返回 exp(x) - 1 的值

参数：
- `x:Var_like` 输入变量

返回：exp(x) - 1的值

返回类型：`Var`

示例：

```python
>>> expr.expm1([9., 0.5])
array([8.1014492e+03, 6.4869785e-01])
```

---
### `add(x, y)`
返回两个输入数的和

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：x+y的值

返回类型：`Var`

示例：

```python
>>> expr.add([9., 0.5], [1.2, -3.0])
array([10.2, -2.5])
```

---
### `subtract(x, y)`
返回两个输入数的差值

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：x - y 的值

返回类型：`Var`

示例：

```python
>>> expr.subtract([9., 0.5], [1.2, -3.0])
array([7.8, 3.5])
```

---
### `multiply(x, y)`
返回两个输入数的乘积

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：x * y 的值

返回类型：`Var`

示例：

```python
>>> expr.multiply([9., 0.5], [1.2, -3.0])
array([10.8, -1.5])
```

---
### `divide(x, y)`
返回两个输入数相除的值，y值不能为0

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：x / y 的值

返回类型：`Var`

示例：

```python
>>> expr.divide([9., 0.5], [1.2, -3.0])
array([7.4999995, -0.16666667])
```

---
### `floordiv(x, y)`
返回小于或等于代数商的最大(最接近正无穷大)int值

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：x // y 的值

返回类型：`Var`

示例：

```python
>>> expr.floordiv([9., 0.5], [1.2, -3.0])
array([7., -1.])
```

---
### `mod(x, y)`
返回两个输入数求余的值，y不能为0，取余运算在计算商值向0方向舍弃小数位

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：x % y 的值

返回类型：`Var`

示例：

```python
>>> expr.mod([9., 0.5], [1.2, -3.0])
array([0.59999967, 0.5])
```

---
### `floormod(x, y)`
返回两个输入数取模的值，y不能为0，取模运算在计算商值向负无穷方向舍弃小数位

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：(1)与y符号相同
     (2)x > y：结果的绝对值与 % 运算相同;
     (3)x < y：①符号相同 结果的绝对值为 y – x ；②符号不同 结果的绝对值与 % 运算相同

返回类型：`Var`

示例：

```python
>>> expr.floormod([9., 0.5], [1.2, -3.0])
array([0.5999994, -2.5])
```

---
### `pow(x, y)`
返回x的y次方的值

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：x的y次方的值

返回类型：`Var`

示例：

```python
>>> expr.pow([9., 0.5], [1.2, -3.0])
array([13.966612, 8.])
```

---
### `minimum(x, y)`
返回输入数中的最小值

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：x和y中的最小值

返回类型：`Var`

示例：

```python
>>> expr.minimum([9., 0.5], [1.2, -3.0])
array([1.2, -3.])
```

---
### `maximum(x, y)`
返回输入数中的最大值

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：x和y中的最大值

返回类型：`Var`

示例：

```python
>>> expr.maximum([9., 0.5], [1.2, -3.0])
array([9., 0.5])
```

---
### `equal(x, y)`
判断输入数是否相等，相等返回1，不等返回0

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：1或0

返回类型：`Var`

示例：

```python
>>> expr.equal([-9., 0.5], [1.2, 0.5])
array([0, 1])
```

---
### `not_equal(x, y)`
判断输入数是否相等，相等返回0，不等返回1

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：1或0

返回类型：`Var`

示例：

```python
>>> expr.not_equal([-9., 0.5], [1.2, 0.5])
array([1, 0])
```

---
### `greater(x, y)`
判断输入数x和y的大小，如果x > y返回1，否者返回0

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：1或0

返回类型：`Var`

示例：

```python
>>> expr.greater([-9., 0.5], [1.2, -3.0])
array([0, 1])
```

---
### `greater_equal(x, y)`
判断输入数x和y的大小，如果x >= y返回1，否者返回0

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：1或0

返回类型：`Var`

示例：

```python
>>> expr.greater_equal([-9., 0.5], [1.2, -3.0])
array([0, 1])
```

---
### `less(x, y)`
判断输入数x和y的大小，如果x < y返回1，否者返回0

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：1或0

返回类型：`Var`

示例：

```python
>>> expr.less([-9., 0.5], [1.2, -3.0])
array([1, 0])
```

---
### `less_equal(x, y)`
判断输入数x和y的大小，如果x <= y返回1，否者返回0

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：1或0

返回类型：`Var`

示例：

```python
>>> expr.less_equal([-9., 0.5], [1.2, -3.0])
array([1, 0])
```

---
### `squared_difference(x, y)`
返回输入数x和输入数y的差的平方值

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：(x - y)^2

返回类型：`Var`

示例：

```python
>>> expr.squared_difference([-9., 0.5], [1.2, -3.0])
array([104.03999, 12.25])
```

---
### `atan2(x, y)`
返回 x / y 的反正切值，别名arctan2

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量

返回：arctan(x / y)的值

返回类型：`Var`

示例：

```python
>>> expr.atan2([-9., 0.5], [1.2, -3.0])
array([-1.4382448, -0.16514869])
```

---
### `logical_or(x, y)`
返回输入数的“或”逻辑的值，x和y形状相同，返回1，否则返回0

参数：
- `x:Var_like` 输入变量，仅支持int32
- `y:Var_like` 输入变量，仅支持int32

返回：1或0

返回类型：`Var`

示例：

```python
>>> expr.logical_or([2, 1], [4, 2])
array([1, 1])
```

---
### `bias_add(value, bias)`
这(主要)是加法的一个特殊情况，其中偏差限制在1-D，value可以有任意数量的维度，与加法不同，在量子化的情况下，偏差的类型允许与值不同。

参数：
- `value` 输入变量，dtype.float或者dtype.int类型
- `bias` 输入变量，一个一维变量，其大小与值的通道维度相匹配，必须与值的类型相同，除非值是量化类型，在这种情况下可以使用不同的量化类型。

返回：与value类型相同的变量

返回类型：`Var`

示例：

```python
 >>> expr.bias_add(np.eye(3,3), np.ones(3))
array([[2., 1., 1.],
       [1., 2., 1.],
       [1., 1., 2.]], dtype=float32)
```

---
### `bitwise_and(x, y)`
返回输入数的“与”运算的值（x & y，x, y必须为dtype.int32

参数：
- `x:Var_like` 输入变量，仅支持int32
- `y:Var_like` 输入变量，仅支持int32

返回：1或0

返回类型：`Var`

示例：

```python
>>> expr.bitwise_and([1, 2], [3, 4])
array([1, 0])
```

---
### `bitwise_or(x, y)`
返回输入数的“或”运算的值（x | y），x, y必须为dtype.int32

参数：
- `x:Var_like` 输入变量，仅支持int32
- `y:Var_like` 输入变量，仅支持int32

返回：x | y

返回类型：`Var`

示例：

```python
>>> expr.bitwise_or([1, 2], [3, 4])
array([3, 6])
```

---
### `bitwise_xor(x, y)`
返回输入数的“异或”运算的值（x ^ y），x, y必须为dtype.int32

参数：
- `x:Var_like` 输入变量，仅支持int32
- `y:Var_like` 输入变量，仅支持int32

返回：x ^ y

返回类型：`Var`

示例：

```python
>>> expr.bitwise_xor([1, 2], [3, 4])
array([2, 6])
```

---
### `reduce_sum(x, axis=[], keepdims=False)`
计算张量x沿着指定的数轴（x的某一维度）上的和

参数：
- `x:Var_like` 输入变量
- `axis : axis_like` 仅支持int32，默认为[],指定按哪个维度进行加和，默认将所有元素进行加和
- `keepdims : bool` 默认为false，表示不维持原来张量的维度，反之维持原张量维度

返回：x沿着指定的数轴（x的某一维度）上的和

返回类型：`Var`

示例：

```python
>>> expr.reduce_sum([[1.,2.],[3.,4.]])
array(10.)
>>> expr.reduce_sum([[1.,2.],[3.,4.]], 0)
array([4., 6.])
```

---
### `reduce_mean(x, axis=[], keepdims=False)`
计算张量x沿着指定的数轴（x的某一维度）上的平均值

参数：
- `x:Var_like` 输入变量
- `axis : axis_like` 仅支持int32，默认为[],指定按哪个维度进行加和，默认将所有元素进行加和
- `keepdims : bool` 默认为false，表示不维持原来张量的维度，反之维持原张量维度

返回：x沿着指定的数轴（x的某一维度）上的平均值

返回类型：`Var`

示例：

```python
>>> expr.reduce_mean([[1.,2.],[3.,4.]])
array(2.5.)
>>> expr.reduce_mean([[1.,2.],[3.,4.]], 0)
array([2., 3.])
```

---
### `reduce_max(x, axis=[], keepdims=False)`
计算x沿着指定的数轴（x的某一维度）上的最大值

参数：
- `x:Var_like` 输入变量
- `axis : axis_like` 仅支持int32，默认为[],指定按哪个维度进行加和，默认将所有元素进行加和
- `keepdims : bool` 默认为false，表示不维持原来张量的维度，反之维持原张量维度

返回：x沿着指定的数轴（x的某一维度）上的最大值

返回类型：`Var`

示例：

```python
>>> expr.reduce_max([[1.,2.],[3.,4.]])
array(4.)
>>> expr.reduce_max([[1.,2.],[3.,4.]], 0)
array([3., 4.])
```

---
### `reduce_min(x, axis=[], keepdims=False)`
计算x沿着指定的数轴（x的某一维度）上的最小值

参数：
- `x:Var_like` 输入变量
- `axis : axis_like` 仅支持int32，默认为[],指定按哪个维度进行加和，默认将所有元素进行加和
- `keepdims : bool` 默认为false，表示不维持原来张量的维度，反之维持原张量维度

返回：x沿着指定的数轴（x的某一维度）上的最小值

返回类型：`Var`

示例：

```python
>>> expr.reduce_min([[1.,2.],[3.,4.]])
array(1.)
>>> expr.reduce_min([[1.,2.],[3.,4.]], 0)
array([1., 2.])
```

---
### `reduce_prod(x, axis=[], keepdims=False)`
计算x沿着指定的数轴（x的某一维度）上的乘积

参数：
- `x:Var_like` 输入变量
- `axis : axis_like` 仅支持int32，默认为[],指定按哪个维度进行加和，默认将所有元素进行加和
- `keepdims : bool` 默认为false，表示不维持原来张量的维度，反之维持原张量维度

返回：x沿着指定的数轴（x的某一维度）上的乘积

返回类型：`Var`

示例：

```python
>>> expr.reduce_prod([[1.,2.],[3.,4.]])
array(24.)
>>> expr.reduce_prod([[1.,2.],[3.,4.]], 0)
array([3., 8.])
```

---
### `reduce_any(x, axis=[], keepdims=False)`
计算x沿着指定的数轴（x的某一维度）上的“逻辑或”的值

参数：
- `x:Var_like` 输入变量
- `axis : axis_like` 仅支持int32，默认为[],指定按哪个维度进行加和，默认将所有元素进行加和
- `keepdims : bool` 默认为false，表示不维持原来张量的维度，反之维持原张量维度

返回：x沿着指定的数轴（x的某一维度）上的“逻辑或”的值

返回类型：`Var`

示例：

```python
>>> expr.reduce_any([[0,1],[0,3]])
array(1)
>>> expr.reduce_any([[0,1],[0,3]], 1)
array([0, 1])
```

---
### `reduce_all(x, axis=[], keepdims=False)`
计算x沿着指定的数轴（x的某一维度）上的“逻辑和”的值

参数：
- `x:Var_like` 输入变量
- `axis : axis_like` 仅支持int32，默认为[],指定按哪个维度进行加和，默认将所有元素进行加和
- `keepdims : bool` 默认为false，表示不维持原来张量的维度，反之维持原张量维度

返回：x沿着指定的数轴（x的某一维度）上的“逻辑和”的值

返回类型：`Var`

示例：

```python
>>> expr.reduce_all([[0,1],[0,3]])
array(0)
>>> expr.reduce_all([[0,1],[0,3]], 0)
array([0, 1])
```

---
### `eltwise_prod(x, y, coeff)`
逐元素对输入的变量执行乘法运算

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量
- `coeff:[float]` 系数，目前仅支持`[1.,0.]`或`[]/[0.]`

返回：`x*y`, 当`coeff=[1.,0.]`时返回`x`

返回类型：`Var`

示例：

```python
>>> expr.eltwise_prod([1., 2., 3.], [2., 2., 2.], [])
array([2., 4., 6.], dtype=float32)
>>> expr.eltwise_prod([1., 2., 3.], [2., 2., 2.], [1., 0.])
array([1., 2., 3.], dtype=float32)
```

---
### `eltwise_sum(x, y, coeff)`
逐元素对输入的变量执行加法运算

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量
- `coeff:[float]` 系数，目前仅支持`[1.,0.]`或`[]/[0.]`

返回：`x+y`, 当`coeff=[1.,0.]`时返回`x`

返回类型：`Var`

示例：

```python
>>> expr.eltwise_sum([1., 2., 3.], [2., 2., 2.], [])
array([3., 4., 5.], dtype=float32)
>>> expr.eltwise_sum([1., 2., 3.], [2., 2., 2.], [1., 0.])
array([1., 2., 3.], dtype=float32)
```

---
### `eltwise_sub(x, y, coeff)`
逐元素对输入的变量执行减法运算

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量
- `coeff:[float]` 系数，目前仅支持`[1.,0.]`或`[]/[0.]`

返回：`x-y`, 当`coeff=[1.,0.]`时返回`x`

返回类型：`Var`

示例：

```python
>>> expr.eltwise_sub([1., 2., 3.], [2., 2., 2.], [])
array([-1.,  0.,  1.], dtype=float32)
>>> expr.eltwise_sub([1., 2., 3.], [2., 2., 2.], [1., 0.])
array([1., 2., 3.], dtype=float32)
```

---
### `eltwise_max(x, y, coeff)`
逐元素对输入的变量执行比较运算，取最大值

参数：
- `x:Var_like` 输入变量
- `y:Var_like` 输入变量
- `coeff:[float]` 系数，目前仅支持`[1.,0.]`或`[]/[0.]`

返回：`max(x,y)`, 当`coeff=[1.,0.]`时返回`x`

返回类型：`Var`

示例：

```python
>>> expr.eltwise_max([1., 2., 3.], [2., 2., 2.], [])
array([2., 2., 3.], dtype=float32)
>>> expr.eltwise_max([1., 2., 3.], [2., 2., 2.], [1., 0.])
array([1., 2., 3.], dtype=float32)
```

---
### `cast(x, dtype=_F.float)`
返回输入数的dtype

参数：
- `x:Var_like` 输入变量
- `dtype` 默认是float类型

返回：x的dtype

返回类型：`Var`

示例：

```python
>>> expr.cast([[0,1],[0,3]], float)
array([[0., 1.],
       [0., 3.]], dtype=float32)
```

---
### `matmul(a, b, transposeA=False, transposeB=False)`
返回两个数组的矩阵乘积

参数：
- `a:Var_like` 输入变量
- `b:Var_like` 输入变量
- `transposeA` 布尔值，用于判定a是否为转置矩阵，默认为false
- `transposeB` 布尔值，用于判定b是否为转置矩阵，默认为false

返回：a @ b 的值，dtype是float32

返回类型：`Var`

示例：

```python
>>> expr.matmul([[1,2],[3,4]], [[1,1],[2,2]])
array([[0., 1.],
       [0., 3.]], dtype=float32)
```

---
### `normalize(x, acrossSpatial, channelShared, eps, scale)`
返回x数据转换成指定的标准化的格式

参数：
- `a:Var_like` 输入变量
- `acrossSpatial` 输入变量，int类型
- `channelShared`  输入变量，int类型
- `eps` 输入变量，float类型，data_format
- `scale` 输入变量，[float]

返回：x数据转换成指定的标准化的格式

返回类型：`Var`

示例：

```python
>>> x = expr.const([-1.0, -2.0, 3.0, 4.0], [1, 2, 2, 1], expr.NCHW)
>>> expr.normalize(x, 0, 0, 0.0, [0.5, 0.5])
array([[[[-0.2236068],
         [-0.4472136]],
        [[ 0.3      ],
         [ 0.4      ]]]], dtype=float32)
```

---
### `argmax(x, axis=0)`
返回数据最大值的索引，根据axis判断返回的是行还是列的最大值索引

参数：
- `x:Var_like` 输入变量
- `axis` 输入变量，int类型，默认为0，当值为0时则按照每一列操作，为1时按照每一行操作

返回：x最大值的索引

返回类型：`Var`

示例：

```python
>>> expr.argmax([[1,2],[3,4]])
array([1, 1], dtype=int32)
```

---
### `argmin(x, axis=0)`
返回数据最小值的索引，根据axis判断返回的是行还是列的最小值索引

参数：
- `x:Var_like` 输入变量
- `axis` 输入变量，int类型，默认为0，0代表列，1代表行

返回：x最小值的索引

返回类型：`Var`

示例：

```python
>>> expr.argmax([[1,2],[3,4]])
array([1, 1], dtype=int32)
```

---
### `cumsum(x, axis=0)`
返回给定axis上的累计和，如果axis == 0，行一不变，行二累加，以此类推；axis == 1，列一不变，列二累加，以此类推

参数：
- `x:Var_like` 输入变量
- `axis` 输入变量，int类型，默认为0，0代表列，1代表行

返回：x在给定axis上的累计和

返回类型：`Var`

示例：

```python
>>> expr.cumsum([[1,2],[3,4]])
array([[1, 2],
       [4, 6]], dtype=int32)
```

---
### `cumprod(x, axis=0)`
返回给定axis上的累计乘积，如果axis == 0，行一不变，行二累计相乘，以此类推；axis == 1，列一不变，列二累计相乘，以此类推

参数：
- `x:Var_like` 输入变量
- `axis` 输入变量，int类型，默认为0，0代表列，1代表行

返回：x在给定axis上的累计乘积

返回类型：`Var`

示例：

```python
>>> expr.cumprod([[1.,2.],[3.,4.]])
array([[1., 2.],
       [3., 8.]], dtype=float32)
```

---
### `svd(x)`
返回'x'的svd矩阵，'x'是一个2D矩阵。
返回'w'，'u'，'vt'为 a = u @ w @ vt。

参数：
- `x:Var_like` 输入变量

返回：w : 形状是 (N).
     u : 形状是 (M, N).
     vt : 形状是 (N, N).

返回类型：`Var`

示例：

```python
>>> expr.cumprod([[1.,2.],[3.,4.]])
[array([5.464986  , 0.36596605], dtype=float32), array([[ 0.40455356,  0.91451436],
       [ 0.91451424, -0.40455365]], dtype=float32), array([[ 0.5760485 ,  0.81741554],
       [-0.81741554,  0.5760485 ]], dtype=float32)]
```

---
### `unravel_index(indices, shape)`
求出数组某元素（或某组元素）拉成一维后的索引值在原本维度（或指定新维度）中对应的索引

参数：
- `indices:Var_like` 输入变量，int32构成的数组， 其中元素是索引值
- `shape:Var_like` 输入变量，一般是原本数组的维度，也可以给定的新维度

返回：索引

返回类型：`Var`

示例：

```python
>>> expr.unravel_index([22, 41, 37], (7,6))
array([[3, 6, 6],
       [4, 5, 1]], dtype=int32)
```

---
### `scatter_nd(indices, updates, shape)`
根据indices将updates散布到新的（初始为零）张量

参数：
- `indices:Var_like` 输入变量，指数张量
- `updates:Var_like` 输入变量，分散到输出的更新
- `shape:Var_like` 输入变量，得到的张量的形状

返回：（初始为零）张量

返回类型：`Var`

示例：

```python
>>> indices = expr.const([4, 3, 1, 7], [4, 1], expr.NHWC, expr.int)
>>> expr.scatter_nd(indices, [9.0, 10.0, 11.0, 12.0], [8])
array([ 0., 11.,  0., 10.,  9.,  0.,  0., 12.], dtype=float32)
```

---
### `one_hot(indices, depth, onValue=1.0, offValue=0.0, axis=-1)`
转换成one_hot类型的张量输出

参数：
- `indices: Var_like` 输入变量，指示on_value的位置，不指示的地方都为off_value。indices可以是向量、矩阵。
- `depth: int` 输入变量，输出张量的尺寸，indices中元素默认不超过（depth-1），如果超过，输出为[0,0,···,0]
- `onValue: float` 输入变量，定义在indices[j] = i 时填充输出值的标量，默认为1
- `offValue: float` 输入变量，定义在indices[j] != i 时填充输出值的标量，默认为0
- `axis: int` 输入变量，要填充的轴，默认为-1，即一个新的最内层轴

返回：one_hot类型的张量输出

返回类型：`Var`

示例：

```python
>>> expr.one_hot([0, 1, 2], 3)
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]], dtype=float32)
```

---
### `broadcast_to(input, shape)`
利用广播将原始矩阵成倍增加

参数：
- `input: Var_like` 输入变量，一个张量，广播的张量
- `shape: int` 输入变量，一个一维的int32张量，期望输出的形状

返回：成倍增加的原始矩阵

返回类型：`Var`

示例：

```python
>>> expr.broadcast_to([1,2], 4)
array([1, 2, 1571999095, -536868871], dtype=int32)
```

---
### `placeholder(shape=[], format=_F.NCHW, dtype=_F.float)`
构建一个Var类型的占位符

参数：
- `shape: [int]` 输入变量，默认是[]
- `format: data_format` 输入变量，默认是NCHW
- `dtype: dtype` 输入变量，默认是float

返回：Var类型的占位符，使用之前需先`write`

返回类型：`Var`

示例：

```python
>>> expr.placeholder()
array(1., dtype=float32)
>>> expr.placeholder([2,2])
array([[1.67e-43, 1.60e-43],
       [1.36e-43, 1.54e-43]], dtype=float32)
```

---
### `clone(x, deepCopy=False)`
克隆一个`Var`

参数：
- `x: Var_like` 输入变量
- `deepCopy: bool` 输入变量，true代表深拷贝，false代表浅拷贝，默认为false

返回：通过x拷贝一个Var

返回类型：`Var`

示例：

```python
>>> x = expr.const([1.,2.], [2])
>>> expr.clone(x, True)
array([1., 2.], dtype=float32)
```

---
### `conv2d(input, weight, bias, stride=(1,1), padding=(0,0), dilate=(1,1), group=1, padding_mode=_F.VALID)`
对由多个输入平面组成的输入信号进行二维卷积

参数：
- `input: var_like` 输入变量，data_format为NC4HW4，输入图像通道数
- `weight: var_like` 输入变量，卷积产生的通道数
- `bias: var_like` 输入变量，在输出中添加一个可学习的偏差
- `stride: tuple of int` 输入变量，卷积步长，默认为(1,1)
- `padding: tuple of int` 输入变量，填充操作，控制padding_mode的数目
- `dilate: tuple of int` 输入变量，扩张操作：控制kernel点（卷积核点）的间距，默认值为(1,1)
- `group: int` 输入变量，控制分组卷积，默认不分组，为1组
- `padding_mode:  Padding_Mode` 输入变量，填充模式，默认是VALID

返回：二维卷积

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 18., 1.), [1, 2, 3, 3])
>>> x = expr.convert(x, expr.NC4HW4)
>>> w = expr.reshape(expr.range(0., 16., 1.), [2, 2, 2, 2])
>>> b = expr.const([1.0, 1.0], [2])
>>> expr.convert(expr.conv2d(x, w, b), expr.NCHW)
array([[[[ 269.,  297.],
         [ 353.,  381.]],
        [[ 685.,  777.],
         [ 961., 1053.]]]], dtype=float32)
```

---
### `conv2d_transpose(input, weight, bias, stride=(1,1), padding=(0,0), dilate=(1,1), group=1, padding_mode=_F.VALID)`
转置卷积，是卷积的反向操作

参数：
- `input: var_like` 输入变量，需要做反卷积的输入图像
- `weight: var_like` 输入变量，反卷积产生的通道数
- `bias: var_like` 输入变量，在输出中添加一个可学习的偏差
- `stride: tuple of int` 输入变量，反卷积步长，默认为(1,1)
- `padding: tuple of int` 输入变量，填充操作，控制padding_mode的数目
- `dilate: tuple of int` 输入变量，扩张操作：控制kernel点（反卷积核点）的间距，默认值为(1,1)
- `group: int` 输入变量，控制分组反卷积，默认不分组，为1组
- `padding_mode:  Padding_Mode` 输入变量，填充模式，默认是VALID

返回：转置卷积（反卷积）

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 18., 1.), [1, 2, 3, 3])
>>> x = expr.convert(x, expr.NC4HW4)
>>> w = expr.reshape(expr.range(0., 16., 1.), [2, 2, 2, 2])
>>> b = expr.const([1.0, 1.0], [2])
>>> expr.convert(expr.conv2d_transpose(x, w, b), expr.NCHW)
array([[[[ 73., 162., 180., 102.],
         [187., 417., 461., 259.],
         [247., 549., 593., 331.],
         [163., 358., 384., 212.]],
        [[109., 242., 276., 154.],
         [283., 625., 701., 387.],
         [391., 853., 929., 507.],
         [247., 534., 576., 312.]]]], dtype=float32)
```

---
### `max_pool(x, kernel, stride, padding_mode=_F.VALID, pads=(0,0))`
最大值池化操作

参数：
- `x: var_like` 输入变量，池化的输入
- `kernel: var_like` 输入变量
- `stride: tuple of int` 输入变量，窗口在每一个维度上滑动的步长
- `padding_mode:  Padding_Mode` 输入变量，填充模式，默认是VALID
- `pads: tuple of int` 输入变量

返回：最大池化

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 18., 1.), [1, 2, 3, 3])
>>> expr.max_pool(x, [2,2], [1,1])
array([[[[ 4.,  5.],
         [ 7.,  8.]],
        [[13., 14.],
         [16., 17.]]]], dtype=float32)
```

---
### `avg_pool(x, kernel, stride, padding_mode=_F.VALID, pads=(0,0))`
平均池化操作

参数：
- `x: var_like` 输入变量，池化的输入
- `kernel: var_like` 输入变量
- `stride: tuple of int` 输入变量，窗口在每一个维度上滑动的步长
- `padding_mode:  Padding_Mode` 输入变量，填充模式，默认是VALID
- `pads: tuple of int` 输入变量

返回：平均池化

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 18., 1.), [1, 2, 3, 3])
    >>> expr.avg_pool(x, [2,2], [1,1])
array([[[[ 2.,  3.],
         [ 5.,  6.]],
        [[11., 12.],
         [14., 15.]]]], dtype=float32)
```

---
### `reshape(x, shape, original_format=_F.NCHW)`
改变输入数的形状

参数：
- `x: var_like` 输入变量，数组或者矩阵
- `shape: axis_like` 输入变量，返回数据的形状
- `original_format : data_format` 输入变量，默认是NCHW

返回：输入数对应的`shape`

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 18., 1.), [1, 2, 3, 3])
>>> reshape(x, [3, 6])
array([[ 0.,  1.,  2.,  3.,  4.,  5.],
       [ 6.,  7.,  8.,  9., 10., 11.],
       [12., 13., 14., 15., 16., 17.]], dtype=float32)
```

---
### `scale(x, channels, scale, bias)`
返回`x * scale + bias`的值

参数：
- `x: var_like` 输入变量
- `channels : int` 输入变量
- `scale : [float]` 输入变量
- `bias : [float]` 输入变量

返回：`x * scale + bias`的值

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 18., 1.), [1, 2, 3, 3])
>>> x = expr.convert(x, expr.NC4HW4)
>>> y = expr.scale(x, 2, [2.0, 1.0], [3.0, 4.0])
>>> expr.convert(y, expr.NCHW)
array([[[[ 3.,  5.,  7.],
           [ 9., 11., 13.],
           [15., 17., 19.]],
          [[13., 14., 15.],
           [16., 17., 18.],
           [19., 20., 21.]]]], dtype=float32)
```

---
### `relu(x, slope=0.0)`
修正线性单元，修正公式为`slope*x if x < 0 else x`

参数：
- `x: var_like` 输入变量
- `slope : float` 输入变量，修正梯度，默认是0.0

返回：修正后的值

返回类型：`Var`

示例：

```python
>>> expr.relu([-1.0, 0.0, 2.0])
    var[0., 0., 2.], dtype=float32)
```

---
### `relu6(x, min=0.0, max=6.0)`
修正线性单元，修正公式为`max(min(x, max), min)`

参数：
- `x: var_like` 输入变量
- `min : float` 输入变量，修正最小范围，默认是0.0
- `max : float` 输入变量，修正最大范围，默认是6.0

返回：修正后的值

返回类型：`Var`

示例：

```python
>>> expr.relu6([-1.0, 7.0, 2.0])
    var[0., 6., 2.], dtype=float32)
```

---
### `prelu(x, slopes)`
查找指定张量输入的泄漏校正线性，修正公式为`slope*x if x < 0 else x`

参数：
- `x: var_like` 输入变量，data_format 是 NC4HW4.
- `slopes : [float]` 输入变量，修正梯度

返回：修正后的值

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(-4., 4., 1.), [1, 2, 2, 2])
>>> x = expr.convert(x, expr.NC4HW4)
>>> expr.convert(expr.prelu(x, [0.5, 0.6]), expr.NCHW)
array([[[[-2. , -1.5],
         [-1. , -0.5]],
        [[ 0. ,  1. ],
         [ 2. ,  3. ]]]], dtype=float32)
```

---
### `softmax(x, axis=-1)`
数学公式:`exp(x)/sum(exp(x), axis)`

参数：
- `x: var_like` 输入变量
- `axis : int` 输入变量，默认为-1

返回：修正后的值

返回类型：`Var`

示例：

```python
>>> expr.softmax([[1., 2.], [3., 4.]], 0)
array([[0.1191897 , 0.1191897 ],
       [0.88081026, 0.88081026]], dtype=float32)
```

---
### `softplus(x)`
通过输入数据 x（张量）并生成一个输出数据（张量）转换公式:`log(exp(x) + 1)`

参数：
- `x: var_like` 输入变量

返回：输出数据（张量）

返回类型：`Var`

示例：

```python
>>> expr.softplus([[1., 2.], [3., 4.]])
array([[1.313261 , 2.1268892],
       [3.048587 , 4.01813  ]], dtype=float32)
```

---
### `softsign(x)`
返回通过转换公式:`x / (abs(x) + 1)`转换后的数据

参数：
- `x: var_like` 输入变量

返回：转换公式`x / (abs(x) + 1)`转换后的数据

返回类型：`Var`

示例：

```python
>>> expr.softsign([[1., 2.], [3., 4.]])
array([[0.5      , 0.6666667],
       [0.75     , 0.8      ]], dtype=float32)
```

---
### `slice(x, starts, sizes)`
返回从张量中提取想要的切片，此操作从由starts指定位置开始的张量x中提取一个尺寸sizes的切片，切片sizes被表示为张量形状，提供公式`x[starts[0]:starts[0]+sizes[0], ..., starts[-1]:starts[-1]+sizes[-1]]`

参数：
- `x: var_like` 输入变量
- `starts : var_like` 输入变量，切片提取起始位置
- `sizes : var_like` 输入变量，切片提取的尺寸

返回：切片数据

返回类型：`Var`

示例：

```python
>>> expr.slice([[1., 2., 3.], [4., 5., 6.]], [0, 1], [1, 2])
array([[2., 3.]], dtype=float32)
```

---
### `split(x, size_splits, axis)`
返回切割张量的子张量数据

参数：
- `x: var_like` 输入变量
- `size_splits : var_like` 输入变量，int类型，切成的份数
- `axis : int` 输入变量，进行切割的维度

返回：切割张量的子张量数据

返回类型：`Var`

示例：

```python
>>> expr.split([[1., 2., 3.], [4., 5., 6.]], [1, 1], 0)
    [array([[1., 2., 3.]], dtype=float32), array([[4., 5., 6.]], dtype=float32)]
```

---
### `strided_slice(x, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0)`
从给定的 x 张量中提取一个尺寸 (end-begin)/stride 的片段，从 begin 片段指定的位置开始，以步长 stride 添加索引，直到所有维度都不小于 end，这里的 stride 可以是负值，表示反向切片,公式：`x[begin[0]:strides[0]:end[0], ..., begin[-1]:strides[-1]:end[-1]]`。

参数：
- `x: var_like` 输入变量
- `begin : var_like` 输入变量，int类型，开始切片处
- `end : var_like` 输入变量，int类型，终止切片处
- `strides : var_like` 输入变量，int类型，步长
- `begin_mask : int` 输入变量，默认为0
- `end_mask : int` 输入变量，默认为0
- `ellipsis_mask : int` 输入变量，默认为0
- `new_axis_mask : int` 输入变量，默认为0
- `shrink_axis_mask : int` 输入变量，默认为0

返回：提取的片段

返回类型：`Var`

示例：

```python
>>> expr.strided_slice([[1., 2., 3.], [4., 5., 6.]], [0, 0], [1, 2], [1, 2])
array([[1.]], dtype=float32)
```

---
### `concat(values, axis)`
根据axis设置的维度拼接输入变量values中的数据

参数：
- `values : [var_like]` 输入变量
- `axis : int` 输入变量，操作的维度

返回：拼接后的数据

返回类型：`Var`

示例：

```python
>>> expr.concat([[1., 2., 3.], [4., 5., 6.]], 0)
array([1., 2., 3., 4., 5., 6.], dtype=float32)
```

---
### `where(x)`
返回满足条件`x > 0`的索引

参数：
- `x : [var_like]` 输入变量

返回：索引，int类型

返回类型：`Var`

示例：

```python
>>> expr.where([0., 1., -2., 3.3])
array([[1],[3]], dtype=int32)
```

---
### `convert(x, format)`
将输入变量 x 的 data_format 转换为 format 

参数：
- `x : [var_like]` 输入变量
- `format : data_format` 输入变量，[NCHW, NHWC, NC4HW4]

返回：转换后的format

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 8., 1.), [1, 2, 2, 2]) # data_format is NCHW
>>> expr.convert(x, expr.NHWC)
array([[[[0., 4.], [1., 5.]],
       [[2., 6.], [3., 7.]]]], dtype=float32)
```

---
### `transpose(x, perm)`
返回一个转置的数据，转置的维度为perm

参数：
- `x : [var_like]` 输入变量
- `perm : [int] or var_like` 新的维度序列

返回：通过perm转置的数据

返回类型：`Var`

示例：

```python
>>> expr.transpose([[1.,2.,3.],[4.,5.,6.]], [1,0])
array([[1., 4.],
       [2., 5.],
       [3., 6.]], dtype=float32)
```

---
### `channel_shuffle(x, axis)`
做以下操作：
    x = _Convert(x, NHWC);
    x = _Reshape(x, {0, 0, 0, group, -1}, NHWC);
    x = _Transpose(x, {0, 1, 2, 4, 3});
    x = _Reshape(x, {0, 0, 0, -1}, NHWC);
    channel_shuffle_res = _Convert(x, NC4HW4);

参数：
- `x : [var_like]` 输入变量
- `axis : int` 输入变量

返回：

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 8., 1.), [1, 4, 1, 2])
>>> expr.convert(expr.channel_shuffle(x, 2), expr.NCHW)
array([[[[0., 1.]],
        [[4., 5.]],
        [[2., 3.]],
        [[6., 7.]]]], dtype=float32)
```

---
### `reverse(x, axis)`
在输入x变量在axis[0]维度进行翻转

参数：
- `x : var_like` 输入变量
- `axis : var_like` 输入变量

返回：反转序列的值

返回类型：`Var`

示例：

```python
>>> expr.reverse(expr.range(-4., 4., 1.), [0])
array([ 3.,  2.,  1.,  0., -1., -2., -3., -4.], dtype=float32)
```

---
### `reverse_sequence(x, y, batch_dim, seq_dim)`
沿着batch_dim维度对x进行切片并反转维度seq_dim上的y[i]元素

参数：
- `x : var_like` 输入变量
- `y : var_like` 输入变量
- `batch_dim : int` 输入变量，切片的维度
- `seq_dim : int` 输入变量，反转的维度

返回：反转序列的值

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 8., 1.), [2, 4])
>>> expr.reverse_sequence(x, [1, 1], 0, 1)
array([[0., 1., 2., 3.],
       [4., 5., 6., 7.]], dtype=float32)
```

---
### `crop(x, size, axis, offset)`
裁剪输入数据x到size对应的尺寸，通过axis维度，起始点offset

参数：
- `x : var_like` 输入变量，data_format 为 NC4HW4.
- `size : var_like` 返回裁剪的尺寸
- `axis : int` 输入变量，裁剪的维度
- `offset : [int]` 输入变量，裁剪的起始点

返回：裁剪后的数据

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 16., 1.), [1, 1, 4, 4])
>>> x = expr.convert(x, expr.NC4HW4)
>>> size = expr.const([0.0, 0.0, 0.0, 0.0], [1, 1, 2, 2])
>>> expr.convert(expr.crop(x, size, 2, [1, 1]), expr.NCHW)
array([[[[ 5.,  6.],
         [ 9., 10.]]]], dtype=float32)
```

---
### `resize(x, x_scale, y_scale)`
调整输入值x的(高度、宽度)的大小，公式为：(y_scale * height, x_scale * width).

参数：
- `x : var_like` 输入变量
- `x_scale : float` 输入变量，图像高度缩放比例
- `y_scale : float` 输入变量，图像宽度缩放比例

返回：(y_scale * height, x_scale * width)的值

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 4., 1.), [1, 1, 2, 2])
>>> x = expr.convert(x, expr.NC4HW4)
>>> expr.convert(expr.resize(x, size, 2, 2), expr.NCHW)
array([[[[0. , 0.5, 1. , 1. ],
         [1. , 1.5, 2. , 2. ],
         [2. , 2.5, 3. , 3. ],
         [2. , 2.5, 3. , 3. ]]]], dtype=float32)
```

---
### `crop_and_resize(x, boxes, box_ind, crop_size, method=_F.BILINEAR, extrapolation_value=0.)`
从输入图像数据中提取固定大小的切片，并指定插值算法 (method) resize成指定的输出大小(crop_size)

参数：
- `x : var_like` 输入变量，data_format 为 NHWC.
- `boxes : var_like` 输入变量，int类型，需要划分的区域
- `box_ind : var_like` 输入变量，是boxes和x之间的索引
- `crop_size : var_like` 输入变量，表示RoiAlign之后的大小
- `method : Interp_Method` 输入变量，[NEAREST, BILINEAR]，默认是BILINEAR，差值算法
- `extrapolation_value : float` 输入变量，默认为0，外插值，boxes>1时裁剪超出图像范围，自动补齐的填充值

返回：crop_size数据

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 16., 1.), [1, 1, 4, 4])
>>> x = expr.convert(x, expr.NHWC)
>>> boxes = expr.const([0.2, 0.3, 0.4, 0.5], [1, 4])
>>> expr.convert(expr.crop_and_resize(x, boxes, [0], [2, 2]), expr.NHWC)
array([[[[3.3000002],
           [3.9      ]],
           [[5.7000003],
           [6.3      ]]]], dtype=float32)
```

---
### `pad(x, paddings, mode=_F.CONSTANT)`
对输入数x某个维度进行填充或复制

参数：
- `x : var_like` 输入变量
- `paddings : var_like` 输入变量，形状是[x.ndim, 2]，int类型，填充的维度
- `mode : PadValue_Mode` 输入变量，[CONSTANT(填充常数), REFLECT(镜像复制), SYMMETRIC(对称复制)]， 默认为CONSTANT

返回：填充或复制后的数据

返回类型：`Var`

示例：

```python
>>> expr.pad([[1.,2.],[3.,4.]], [[0, 1], [1,2]])
array([[0., 1., 2., 0., 0.],
       [0., 3., 4., 0., 0.],
       [0., 0., 0., 0., 0.]], dtype=float32)
```

---
### `randomuniform(shape, dtype=_F.float, low=0.0, high=1.0, seed0=0, seed1=1)`
用于从均匀分布中输出随机值

参数：
- `shape : axis_like` 输入变量，数据形状
- `dtype : dtype` 输出的类型：[float, double, int, int64, uint8]，默认为float
- `low : float` 输入变量，随机值范围下限，默认为0
- `high : float` 输入变量，随机值范围上限，默认为1
- `seed0 : int` 输入变量，随机因子，默认为0
- `seed1 : int` 输入变量，随机因子，默认为0

返回：随机值

返回类型：`Var`

示例：

```python
>>> expr.pad([[1.,2.],[3.,4.]], [[0, 1], [1,2]])
array([[0., 1., 2., 0., 0.],
       [0., 3., 4., 0., 0.],
       [0., 0., 0., 0., 0.]], dtype=float32)
```

---
### `expand_dims(x, axis)`
给输入数据在axis轴上新增一个维度

参数：
- `x : var_like` 输入变量
- `axis : int or var_like` 输入变量，int类型，指定扩大输入数据形状的维度索引值

返回：新增维度后的数据

返回类型：`Var`

示例：

```python
>>> expr.expand_dims([1.,2.], 0)
array([[1., 2.]], dtype=float32)
```

---
### `rank(x)`
返回输入数据的维度

参数：
- `x : var_like` 输入变量

返回：维度信息

返回类型：`Var`

示例：

```python
>>> expr.rank([[1.,2.],[3.,4.]])
array(2, dtype=int32)
```

---
### `size(x)`
返回输入数据的大小

参数：
- `x : var_like` 输入变量

返回：大小size

返回类型：`Var`

示例：

```python
>>> expr.size([[1.,2.],[3.,4.]])
array(4, dtype=int32)
```

---
### `shape(x)`
返回输入数据的形状


参数：
- `x : var_like` 输入变量

返回：形状

返回类型：`Var`

示例：

```python
>>> expr.shape([[1.,2.],[3.,4.]])
array([2, 2], dtype=int32)
```

---
### `stack(values, axis=0)`
将输入数据x和要沿着axis维度进行拼接

参数：
- `x : var_like` 输入变量
- `axis : int` 输入变量，默认为0，指明对矩阵的哪个维度进行拼接

返回：拼接的数据

返回类型：`Var`

示例：

```python
>>> x = expr.const([1., 2.], [2])
>>> y = expr.const([3., 4.], [2])
>>> expr.stack([x, y])
array([[1., 2.],
       [3., 4.]], dtype=float32)
```

---
### `unstack(x, axis=0)`
将输入数据x沿着axis维度堆叠为r-1级数据

参数：
- `x : var_like` 输入变量
- `axis : int` 输入变量，默认为0，指明对矩阵的哪个维度进行分解

返回：分解的数据

返回类型：`Var`

示例：

```python
>>> expr.unstack([[1., 2.], [3., 4.]])
    [array([1., 2.], dtype=float32), array([3., 4.], dtype=float32)]
```

---
### `fill(shape, value)`
创建一个填充有标量值的数据

参数：
- `shape : var_like` 输入变量，int类型，定义输出张量形状的整数数组
- `value : var_like` 输入变量，一个标量值，用于填充输出数据。

返回：填充有标量值的指定形状的数据

返回类型：`Var`

示例：

```python
>>> expr.fill([2,3], 1.0)
array([[1., 1., 1.],
       [1., 1., 1.]], dtype=float32)
```

---
### `tile(x, multiples)`
返回输入数x在同一维度上重复multiples次的数据

参数：
- `x : var_like` 输入变量
- `multiples : var_like` 输入变量，int类型，同一维度上重复的次数

返回：x重复multiples次的数据

返回类型：`Var`

示例：

```python
>>> expr.tile([2,3], [3])
array([2, 3, 2, 3, 2, 3], dtype=int32)
```

---
### `gather(x, index)`
根据下标返回对应的值

参数：
- `x : var_like` 输入变量
- `index : var_like` 输入变量，int类型，下标索引

返回：`x[index]`的值

返回类型：`Var`

示例：

```python
>>> expr.gather([[1.,2.],[3.,4.]], [1])
array([[3., 4.]], dtype=float32)
```

---
### `gather_nd(x, indices)`
根据indeces描述的索引，在x中提取元素，拼接成一个新的数据

参数：
- `x : var_like` 输入变量
- `indices : var_like` 输入变量，int类型，索引

返回：`x[indices[0], ..., indices[-1]]`

返回类型：`Var`

示例：

```python
>>> expr.gather_nd([[1.,2.],[3.,4.]], [1, 0])
array(3., dtype=float32)
```

---
### `select(cond, x, y)`
返回根据'cond'从'x'或'y'中选择的元素

参数：
- `cond : var_like` 输入变量
- `x : var_like` 输入变量
- `y : var_like` 输入变量

返回：根据条件选中的元素

返回类型：`Var`

示例：

```python
>>> expr.select([1., 0., 2.], [-1., -2., -3.], [1., 2., 3.])
array([-1.,  2., -3.], dtype=float32)
```

---
### `squeeze(x, axes=[])`
删除到指定位置的尺寸为1的新数据

参数：
- `x : var_like` 输入变量
- `axes : axis_like` 输入变量，默认为[]，用来指定要删掉的为1的维度

返回：变换后的数据

返回类型：`Var`

示例：

```python
>>> expr.squeeze([[[1.0, 2.0]]], [0, 1])
array([1., 2.], dtype=float32)
```

---
### `unsqueeze(x, axes=[])`
插入到指定位置的尺寸为1的新数据

参数：
- `x : var_like` 输入变量
- `axes : axis_like` 输入变量，默认为[]，用来指定要增加的为1的维度

返回：变换后的数据

返回类型：`Var`

示例：

```python
>>> expr.unsqueeze([1.0, 2.0], [0, 1])
array([[[1., 2.]]], dtype=float32)
```

---
### `depth_to_space(x, axis)`
将数据从深度重新排列为空间数据块

参数：
- `x : var_like` 输入变量，data_format为NHWC.
- `axis : int` 输入变量，操作的维度

返回：空间数据块

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 12., 1.), [1, 4, 1, 3])
>>> expr.depth_to_space(x, 2)
array([[[[ 0.,  3.,  1.,  4.,  2.,  5.],
         [ 6.,  9.,  7., 10.,  8., 11.]]]], dtype=float32)
```

---
### `space_to_depth(x, axis)`
重新排列空间数据块，进入深度。

参数：
- `x : var_like` 输入变量，data_format为NHWC.
- `axis : int` 输入变量，操作的维度

返回：空间数据块，进入深度

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 16., 1.), [1, 1, 4, 4])
>>> x = expr.convert(x, expr.NHWC)
>>> expr.space_to_depth(x, 2)
array([[[[ 0.,  1.,  4.,  5.],
         [ 2.,  3.,  6.,  7.]],
        [[ 8.,  9., 12., 13.],
         [10., 11., 14., 15.]]]], dtype=float32)
```

---
### `batch_to_space_nd(x, block_shape, crops)`
将给定的 “batch” 从零维重构为具有“block-Shape + [batch]”形状的 “M+1” 维，其中 block-Shape 是参数，batch 是指定的张量。这里根据给定的裁剪数组对 in-between 结果进行裁剪。

参数：
- `x : var_like` 输入变量，data_format为NC4HW4.
- `block_shape : var_like` 输入变量，int类型，一维数组的形状必须为 [M]，因为所有值都必须大于或等于 1
- `crops : var_like` 输入变量，int类型，[M, 2] 的二维数组形状，其​​中所有值必须大于或等于0，这里的crop[i] = [cropStart,cropEnd] 定义了从输入维度 i + 1 裁剪的部分

返回：指定批次的重塑版本的数据

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 12., 1.), [4, 1, 1, 3])
>>> x = expr.convert(x, expr.NC4HW4)
>>> crops = expr.const([0, 0, 0, 0], [2, 2], expr.NCHW, expr.int)
>>> expr.convert(expr.batch_to_space_nd(x, [2, 2], crops), expr.NHWC)
array([[[[ 0.],
         [ 3.],
         [ 1.],
         [ 4.],
         [ 2.],
         [ 5.]],
        [[ 6.],
         [ 9.],
         [ 7.],
         [10.],
         [ 8.],
         [11.]]]], dtype=float32)
```

---
### `space_to_batch_nd(x, block_shape, paddings)`
将指定输入空间的空间维度拆分为块的矩阵，其形状为 blockShape，其中 blockshape 是参数。这里根据指定的填充数组填充空间维度

参数：
- `x : var_like` 输入变量，data_format为NC4HW4.
- `block_shape : var_like` 输入变量，int类型，它是一维数组，必须具有 [M] 的形状，因为所有值必须大于或等于 1
- `paddings : var_like` 输入变量，int类型，一个形状为 [M, 2] 的二维数组，所有值必须大于或等于0。这里 padding 等于 [padStart, padEnd]

返回：指定输入空间的拆分版本的数据

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 12., 1.), [3, 1, 2, 2])
>>> x = expr.convert(x, expr.NC4HW4)
>>> paddings = expr.const([0, 0, 0, 0], [2, 2], expr.NCHW, expr.int)
>>> expr.convert(expr.space_to_batch_nd(x, [2, 2], paddings), expr.NHWC)
array([[[[ 0.]]],
       [[[ 4.]]],
       [[[ 8.]]],
       [[[ 1.]]],
       [[[ 5.]]],
       [[[ 9.]]],
       [[[ 2.]]],
       [[[ 6.]]],
       [[[10.]]],
       [[[ 3.]]],
       [[[ 7.]]],
       [[[11.]]]], dtype=float32)
```

---
### `elu(x, alpha)`
激励函数，主要是为计算图归一化返回结果而引进的非线性部分

参数：
- `x : var_like` 输入变量
- `alpha : float` 输入变量，预定义常量

返回：

返回类型：`Var`

示例：

```python
>>> expr.elu([-1.0, 2.0], 1.673263)
array([-1.0577048,  2.], dtype=float32)
```

---
### `selu(x, scale, alpha)`
比例指数线性单元激活函数

参数：
- `x : var_like` 输入变量
- `scale : float` 输入变量，预定义常量
- `alpha : float` 输入变量，预定义常量

返回：缩放的指数单位激活： scale * elu(x, alpha).

返回类型：`Var`

示例：

```python
>>> expr.selu([-1.0, 2.0], 1.0507, 1.673263)
array([-1.1113304, 2.1014], dtype=float32)
```

---
### `matrix_band_part(x, num_lower, num_upper)`
计算矩阵维度

参数：
- `x : var_like` 输入变量
- `num_lower : var_like` 输入变量，int类型，要保持的对角线的数量，如果为负,则保留整个下三角
- `num_upper : var_like` 输入变量，int类型，要保留的superdiagonals数，如果为负,则保持整个上三角.

返回：返回一个与 x 具有相同的类型且有相同形状的秩为k的数据，提取的带状数据

返回类型：`Var`

示例：

```python
>>> expr.matrix_band_part([[-2., 1.], [-1., 2.]], 1, -1)
array([[-2.,  1.],
       [-1.,  2.]], dtype=float32)
```

---
### `moments(x, axes=[2, 3], shift=None, keep_dims=True)`
用于在指定维度计算均值和方差

参数：
- `x : var_like` 输入变量，data_format为NC4HW4
- `axes : axis_like` 输入变量，int类型，指定计算均值和方差的轴，默认是[2,3]
- `shift : var_like` 输入变量，int类型，未在当前实现中使用，默认是None
- `keep_dims : bool` 输入变量，保持维度，表示产生是否与输入数据具有相同的维度，默认是true

返回：均值和方差

返回类型：`Var`

示例：

```python
>>> x = expr.reshape(expr.range(0., 4., 1.), [1, 1, 2, 2])
>>> x = expr.convert(x, expr.NC4HW4)
>>> expr.moments(x)
[array([[[[1.5]]]], dtype=float32), array([[[[1.25]]]], dtype=float32)]
```

---
### `setdiff1d(x, y)`
返回在x中出现，但是在y中没有出现的的元素

参数：
- `x : var_like` 输入变量
- `y : var_like` 输入变量

返回：在x中出现，但是在y中没有出现的的元素

返回类型：`Var`

示例：

```python
>>> expr.setdiff1d([1, 2, 3], [2, 3, 4])
array([1], dtype=int32)
```

---
### `zeros_like(x)`
把所有元素都置为零

参数：
- `x : var_like` 输入变量

返回：所有元素都为零的张量

返回类型：`Var`

示例：

```python
>>> expr.zeros_like([[1, 2], [3, 4]])
array([[0, 0],
       [0, 0]], dtype=int32)
```

---
### `range(start, limit, delta)`
创建数字序列变量

参数：
- `start : var_like` 输入变量，int类型，表示范围的起始编号
- `limit : var_like` 输入变量，int类型，最大编号限制，并且不包括在内
- `delta : var_like` 输入变量，int类型，增量

返回：数字序列变量

返回类型：`Var`

示例：

```python
>>> expr.range(1.0, 7.0, 2.0)
array([1., 3., 5.], dtype=float32)
```

---
### `sort(x, axis=-1, arg=False, descend=False)`
排序

参数：
- `x : var_like` 输入变量
- `axis : int` 输入变量，int类型
- `arg : bool` 输入变量，是否返回排序元素的index， 默认为false
- `descend : bool` 输入变量，true代表倒序，false代表正序，默认为false

返回：排序结果

返回类型：`Var`

示例：

```python
>>> expr.sort([[5, 0], [1, 3]])
array([[1, 0],
       [5, 3]], dtype=int32)
```

---
### `nms(boxes, scores, max_detections, iou_threshold=-1.0, score_threshold=-1.0)`
非极大值抑制算法，搜索局部极大值，抑制非极大值元素

参数：
- `boxes : var_like` 输入变量，形状必须为[num, 4]
- `scores : var_like` 输入变量，float类型的大小为[num_boxes]代表上面boxes的每一行，对应的每一个box的一个score
- `max_detections : int` 输入变量，一个整数张量，代表最多可以利用NMS选中多少个边框
- `iou_threshold : float` 输入变量，IOU阙值展示的是否与选中的那个边框具有较大的重叠度，默认为0
- `score_threshold : float` 输入变量，默认为float_min，来决定什么时候删除这个边框

返回：搜索局部极大值，抑制非极大值元素

返回类型：`Var`

示例：

```python
>>> expr.nms([[1, 1, 4, 4], [0, 0, 3, 3], [5, 5, 7, 7]], [0.9, 0.5, 0.1], 3, 0.1)
array([0, 2], dtype=int32)
array([5., 4.5])
```

---
### `raster(vars, region, shape)`
使用`Raster`创建一个映射关系，`Raster`是表示内存映射的元算子；
`region`使用`[int]`来描述；其表示了从`var`到结果的内存映射关系，对应的C++数据结构如下：
```cpp
struct View {
    int32_t offset = 0;
    int32_t stride[3] = {1, 1, 1};
};

struct Region {
    View src;
    View dst;
    int32_t size[3] = {1, 1, 1};
};
```
在Python中使用11个`int`来表示Region，顺序为：`src_offset, src_stride[3], dst_offset, dst_stride[3], size[3]`；多个region则继续增加int数目，总数目应该为11的倍数；并且region的数目应该与vars的数目相等

参数：
- `var : [Var]` 输入变量，内存映射的数据来源
- `region : [int]` 表示内存映射关系
- `shape : [int]` 输出变量的形状

返回：内存映射后的变量

返回类型：`Var`

示例：

```python
>>> var = expr.const([1., 2., 3. ,4.], [2, 2])
>>> expr.raster([var], [0, 4, 1, 2, 0, 4, 2, 1, 1, 2, 2], [2, 2]) # do transpose
array([[1., 3.],
       [2., 4.]], dtype=float32)
```

---
### `quant(var, scale, min=-128, max=127, zero=0)`
量化，根据`scale`把float类型的输入量化为int8类型的输出，量化公式为：`y = clamp(x / scale + zero, min, max)`

参数：
- `var : Var` 输入变量，dtype为`float`, data_format为`NC4HW4`
- `scale : Var` 量化的scale值，dtype为`float`
- `min : int` 输出变量的最小值，默认为-128
- `max : int` 输出变量的最大值，默认为127
- `zero : int` 零点值，默认为0

返回：量化后的int8类型变量

返回类型：`Var`

示例：

```python
>>> x = expr.const([1., 2., 3. ,4.], [4])
>>> x = expr.convert(x, expr.NC4HW4)
>>> expr.quant(x, 0.2, -128, 127)
array([-128, -128, -127, -127], dtype=int8)
```

---
### `dequant(var, scale, zero=0)`
反量化，根据`scale`把int8类型的输入反量化为float类型的输出，量化公式为：`y = (x - zero) * scale`

参数：
- `var : Var` 输入变量，dtype为int8
- `scale : Var` 反量化的scale值，dtype为float
- `zero : int` 反量化的zero值，默认为0

返回：反量化后的float类型变量

返回类型：`Var`

示例：

```python
>>> x = expr.const([-128, -128, -127, -127], [4], NCHW, expr.int8)
>>> x = expr.convert(x, expr.NC4HW4)
>>> expr.dequant(x, 0.2, 0)
array([0. , 0. , 0.2, 0.2], dtype=float32)
```

---
### `histogram(input, binNum, minVal, maxVal)`
计算输入变量在指定范围内的直方图分布

参数：
- `input : var_like` 输入变量
- `binNum : int` 直方图桶的个数
- `minVal : int` 直方图计算的最小值
- `maxVal : int` 直方图计算的最大值

返回：直方图统计结果

返回类型：`Var`

示例：

```python
>>> expr.histogram(expr.range(0., 10.0, 1.0), 5, 1, 9)
array([2., 2., 1., 2., 2.], dtype=float32)
```

---
### `detection_post_process(encode_boxes, class_predictions, anchors, num_classes, max_detections, max_class_per_detection, detections_per_class, nms_threshold, iou_threshold, use_regular_nms, centersize_encoding)`
SSD检测模型后处理函数

参数：
- `encode_boxes : Var` 检测框坐标
- `class_predictions : Var` 分类结果概率
- `anchors : Var` 锚点
- `num_classes : int` 分类个数
- `max_detections : int` 最大检测值
- `max_class_per_detection : int` 每个检测的最大种类
- `detections_per_class : int` 每个类别的检测结果
- `nms_threshold : float` nms阈值
- `iou_threshold : float` iou阈值
- `use_regular_nms : bool` 是否使用常规nms，目前仅支持`False`
- `centersize_encoding : [float]` 中心尺寸编码

返回：后处理结果

返回类型：`Var`

---
### `roi_pooling(input, roi, pooledHeight, pooledWidth, spatialScale, outputGrad, backwardDiff)`
roi_pooling

参数：
- `input : Var` 输入变量，dtype为int8
- `roi : Var` 反量化的scale值，dtype为float
- `pooledHeight : int` 反量化的zero值，默认为0
- `pooledWidth : int` 反量化的zero值，默认为0

返回：roipooling结果

返回类型：`Var`

示例：

```python
TODO
```

---
### `roi_align(input, roi, pooledHeight, pooledWidth, spatialScale, samplingRatio, aligned, poolType, outputGrad, backwardDiff)`

roialign

参数：
- `input : Var` 输入变量，dtype为int8
- `roi : Var` 反量化的scale值，dtype为float
- `pooledHeight : int` pooling的
- `pooledHeight : int` 反量化的zero值，默认为0

返回：roialign结果

返回类型：`Var`

示例：

```python
TODO
```
---
**以下函数为框架开发者使用函数，普通用户不建议使用！**

---
### `load_as_dict(fileName)` *[deprecated]*
从文件中加载模型，并将模型转换为计算图，以`dict`的形式返回计算图的所有节点名称和`Var`

*不建议使用该接口*

参数：
- `fileName:str` 模型文件路径

返回：加载的模型计算图，其`key`为`str`，`value`为`Var`

返回类型：`dict`

示例：

```python
>>> vars = expr.load_as_dict('mobilenet_v1.mnn')
>>> vars.keys()
dict_keys(['conv1', 'conv2_1/dw', 'conv2_1/sep', 'conv2_2/dw', 'conv2_2/sep', 'conv3_1/dw', 'conv3_1/sep', 'conv3_2/dw', 'conv3_2/sep', 'conv4_1/dw', 'conv4_1/sep', 'conv4_2/dw', 'conv4_2/sep', 'conv5_1/dw', 'conv5_1/sep', 'conv5_2/dw', 'conv5_2/sep', 'conv5_3/dw', 'conv5_3/sep', 'conv5_4/dw', 'conv5_4/sep', 'conv5_5/dw', 'conv5_5/sep', 'conv5_6/dw', 'conv5_6/sep', 'conv6/dw', 'conv6/sep', 'data', 'fc7', 'pool6', 'prob'])
```
---
### `get_inputs_and_outputs(allVariable)` *[deprecated]*
获取`dict`形式计算图的输入输出节点，可以在使用V3接口时获取输入输出的信息

参数：
- `allVariable:dict` 计算图的`dict`形式，其`key`为`str`，`value`为`Var`

返回：计算图的输入输出，其中输入输出都为`dict`形式，其`key`为`str`，`value`为`Var`

返回类型：`(dict, dict)`

示例：

```python
>>> vars = expr.load_as_dict('mobilenet_v1.mnn')
>>> inputs, outputs = expr.get_inputs_and_outputs(vars)
>>> inputs.keys()
dict_keys(['data'])
>>> outputs.keys()
dict_keys(['prob'])
```
