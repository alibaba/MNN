## numpy

```python
module numpy
```
numpy模块提供了基础的数值计算函数，在[expr](expr.md)模块基础上提供了兼容了[numpy](https://numpy.org/)的API

*numpy兼容参数是指为了兼容numpy的api，但是在MNN.numpy中无效的参数*

**用法注意：**
- 此numpy模块并非完整支持`numpy`的所有API，当遇到函数或用法不支持时可以尝试用其他函数拼凑；
- 此numpy模块数值计算基于`MNN.expr`封装，计算过程会有创建`Op`的开销，对于较小规模的`Var`计算性能相比`numpy`会有比较明显的劣势，因此尽量避免使用此模块处理小规模数据；
- `Var`的存取可以利用`expr.save`和`expr.load_as_list`实现；

---
### `numpy Types`
- [Var](Var.md)

---
### `numpy.dtype`
描述Var的数据类型，是`expr.dtype`的封装
- 类型：`Enum`
- 枚举值：
  - `uint8`
  - `int32`
  - `int64`
  - `float32`
  - `float64`

---
### `concatenate(args, axis=0, out=None, dtype=None, casting="same_kind")`
作用等同与 `numpy` 中的 [`np.concatenate`](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html) 函数，沿指定轴连接输入的`Var`。

参数：
- `args:Var序列` 被连接的变量
- `axis:int` 指定连接的轴
- `out:numpy兼容参数` 默认为None
- `dtype:numpy兼容参数` 默认为None
- `casting:numpy兼容参数` 默认为None

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> a = np.array([[1, 2], [3, 4]])
>>> b = np.array([[5, 6]])
>>> np.concatenate((a, b), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]], dtype=int32)
```
---
### `pad(array, pad_width, mode='constant')`
作用等同与 `numpy` 中的 [`np.pad`](https://numpy.org/doc/stable/reference/generated/numpy.pad.html) 函数，对输入变量的边沿按照指定规则进行扩充。

参数：
- `array:Var` 将要扩充的变量
- `pad_width:Var，扩充的数目，一共为维度的2倍` 扩充的数目，一共为维度的2倍，分别对应一个维度前后
- `mode:{'constant', 'reflect', 'symmetric'}` 填充的方式

返回：扩充后得到的变量 

返回类型：`Var` 

示例

```python
a = [1, 2, 3, 4, 5]
>>> np.pad(a, (2, 3), 'constant')
array([0, 0, 1, 2, 3, 4, 5, 0, 0, 0], dtype=int32)
>>> np.pad(a, (2, 3), 'reflect')
array([3, 2, 1, 2, 3, 4, 5, 4, 3, 2], dtype=int32)
>>> np.pad(a, (2, 3), 'symmetric')
array([2, 1, 1, 2, 3, 4, 5, 5, 4, 3], dtype=int32)
```
---
### `sign(x)`
作用等同与 `numpy` 中的 [`np.sign`](https://numpy.org/doc/stable/reference/generated/numpy.sign.html) 函数，按位获取符号，正数返回1负数返回-1。

参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.sign([1.,-2.])
array([ 1., -1.], dtype=float32)
```
---
### `asscalar(a)`
作用等同与 `numpy` 中的 [`np.asscalar`](https://numpy.org/doc/stable/reference/generated/numpy.asscalar.html) 函数，将数据转换为scalar。

参数：
- `a:Var` 输出变量的数据来源

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> np.asscalar(np.array([24]))
24
```
---
### `shape(a)`
作用等同与 `numpy` 中的 [`np.shape`](https://numpy.org/doc/stable/reference/generated/numpy.shape.html) 函数，获取输入Var的形状，类型为`tuple`。

参数：
- `a:Var` 将会输出该变量的形状

返回：`Var`的形状 

返回类型：`tuple`

示例

```python
>>> a = np.ones([[2],[2]])
>>> np.shape(a)
(2, 2)
```
---
### `mat(a, dtype=None)`
作用等同与 `numpy` 中的 [`np.mat`](https://numpy.org/doc/stable/reference/generated/numpy.mat.html) 函数，根据指定数据，将输入数据转换为ndim=2的Var。

参数：
- `a:Var|list|tuple` 输出变量的数据来源
- `dtype:dtype` 重新指定输出的类型; 默认为a的数据类型

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> np.mat([1, 2, 3, 4])
array([[1, 2, 3]])
```
---
### `divmod(x1, x2)`
作用等同与 `numpy` 中的 [`np.divmod`](https://numpy.org/doc/stable/reference/generated/numpy.divmod.html) 函数，返回2个输入的(floordiv, mod)值。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`tuple` of `Var` 

示例

```python
>>> np.divmod(2, 3)
(array(0, dtype=int32), array(2, dtype=int32))
```
---
### `zeros(shape, dtype=None, order='C')`
作用等同与 `numpy` 中的 [`np.zeros`](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html) 函数，创建一个指定形状，类型的Var，其全部的值都为0，是`full`的特例。

参数：
- `shape:[int]` 指定输出的形状
- `dtype:dtype` 指定输出的形状类型; 默认为`np.float32`
- `order:numpy兼容参数` 默认为None

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> np.ones((2, 2))
array([[0., 0.],
       [0., 0.]])
```
---
### `matrix(a, dtype=None)`
作用等同与 `numpy` 中的 [`np.matrix`](https://numpy.org/doc/stable/reference/generated/numpy.matrix.html) 根据指定数据，将输入数据转换为ndim=2的Var。

参数：
- `a:Var|list|tuple` 输出变量的数据来源
- `dtype:dtype` 重新指定输出的类型; 默认为a的数据类型

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> np.matrix([1, 2, 3, 4])
array([[1, 2, 3]])
```
---
### `round_(x)`
作用等同与 `numpy` 中的 [`np.round_`](https://numpy.org/doc/stable/reference/generated/numpy.round_.html) 函数，按位进行四舍五入。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.round_([1.2, 3.5, 4.7])
array([1., 4., 5.], dtype=float32)
```
---
### `ones(shape, dtype=None, order='C')`
作用等同与 `numpy` 中的 [`np.ones`](https://numpy.org/doc/stable/reference/generated/numpy.ones.html) 函数，创建一个指定形状，类型的Var，其全部的值都为1，是`full`的特例。

参数：
- `shape:[int]` 指定输出的形状
- `dtype:dtype` 指定输出的形状类型; 默认为`np.float32`
- `order:numpy兼容参数` 默认为None

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> np.ones((2, 2))
array([[1., 1.],
     [1., 1.]])
```
---
### `expand_dims(x, axis)`
作用等同与 `numpy` 中的 [`np.expand_dims`](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html) 函数，扩展输入的形状，在指定位置插入一个新维度，是expr.unsqueeze的封装。

参数：
- `x:Var` 需要扩展维度的变量
- `axis:int`  插入新维度的位置

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> a = np.array([1, 2])
>>> b = np.expand_dims(a, axis=0)
>>> a.shape
[2]
>>> b.shape
[1,2]
```
---
### `rollaxis(a, axis, start=0)`
作用等同与 `numpy` 中的 [`np.rollaxis`](https://numpy.org/doc/stable/reference/generated/numpy.rollaxis.html) 函数，将`Var`的指定维度滚动到目标位置。

参数：
- `a:Var` 将会滚动该变量的维度
- `axis:int` 指定要滚动的维度
- `start:int` 滚动维度的目标位置

返回：滚动维度后的`MNN.Var`

返回类型：`Var`

示例

```python
>>> a = np.ones((3,4,5,6))
>>> np.rollaxis(a, 3, 1).shape
(3, 6, 4, 5)
```
---
### `asarray(a, dtype=None, order=None)`
作用等同与 `numpy` 中的 [`np.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html) 根据指定数据，将输入数据转换为一个Var。

参数：
- `a:ndarray` 输出变量的数据来源
- `dtype:dtype` 重新指定输出的类型; 默认为a的数据类型
- `order:numpy兼容参数` 默认为None

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> np.asarray([1, 2, 3])
array([1, 2, 3])
```
---
### `repeat(a, repeats, axis=None)`
作用等同与 `numpy` 中的 [`np.repeat`](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html) 函数，将输入变量的元素重复指定次数。

参数：
- `A:Var` 被复制的Var
- `repeats:int` 复制的次数
- `axis:numpy兼容参数` 默认为None

返回：复制得到的`Var` 

返回类型：`Var` 

示例

```python
>>> a = np.array([0, 1, 2])
>>> np.repeat(a, 2)
array([0, 0, 1, 1, 2, 2], dtype=int32)
```
---
### `zeros_like(a, dtype=None, order='K', subok=True, shape=None)`
作用等同与 `numpy` 中的 [`np.zeros_like`](https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html) 函数，创建一个与指定Var的形状类型相同的Var，其全部的值为0, 为`full_like`的特例。

参数：
- `a:Var` 输出Var的形状和数据类型与该Var相同
- `dtype:dtype` 指定输出的形状类型; 默认为`np.float32`
- `order:numpy兼容参数` 默认为None
- `subok:numpy兼容参数` 默认为None
- `shape:[int]` 重新指定输出的形状

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> a = np.empty([2,2])
>>> np.ones_like(a)
array([[0., 0.],
     [0., 0.]])
```
---
### `clip(x, a_min, a_max)`
作用等同与 `numpy` 中的 [`np.clip`](https://numpy.org/doc/stable/reference/generated/numpy.clip.html) 函数, 对元素按照最小最大的范围进行约束。

参数：
- `x:Var` 参与计算的变量
- `a_min:scalar` 约束的最小值
- `a_max:scalar` 约束的最大值

返回：计算得到的变量 

返回类型：`tuple` of `Var` 

示例

```python
>>> np.clip(np.array([-3., -1., 3.]), -2., 2.)
array([-2., -1.,  2.], dtype=float32)
```
---
### `sin(x)`
作用等同与 `numpy` 中的 [`np.sin`](https://numpy.org/doc/stable/reference/generated/numpy.sin.html) 函数，按元素计算正弦值。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.sin([1.,2.])
array([0.84147096, 0.9092974 ], dtype=float32)
```
---
### `cos(x)`
作用等同与 `numpy` 中的 [`np.cos`](https://numpy.org/doc/stable/reference/generated/numpy.cos.html) 函数，按元素计算余弦值。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.cos([1.,2.])
array([ 0.5403023 , -0.41614684], dtype=float32)
```
---
### `count_nonzero(a, axis=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.count_nonzero`](https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html) 函数，计算非0元素的数目。

参数：
- `a:Var` 将要计数的变量
- `axis:int` 计数的轴
- `keepdims:bool` 是否保留计数维度

返回：计数得到的变量 

返回类型：`int` of `Var` 

示例

```python
>>> np.count_nonzero(np.eye(4))
4
```
---
### `cbrt(x)`
作用等同与 `numpy` 中的 [`np.cbrt`](https://numpy.org/doc/stable/reference/generated/numpy.cbrt.html) 函数，按元素计算立方根。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.cbrt([1.,2.])
array([1., 1.2599211], dtype=float32)
```
---
### `ptp(a, axis=None)`
作用等同与 `numpy` 中的 [`np.ptp`](https://numpy.org/doc/stable/reference/generated/numpy.ptp.html) 函数，返回沿axis轴的数值范围(最大值减去最小值)。


参数：
- `a:Var` 将要统计的变量
- `axis:int` 统计的轴

返回：计数得到的变量 

返回类型：`int` of `Var` 

示例

```python
>>> a = np.arange(4).reshape((2,2))
>>> np.ptp(a) 
3
>>> np.ptp(a, axis=0) 
array([2, 2], dtype=int32)
```
---
### `where(indices, shape, order='C')`
作用等同与 `numpy` 中的 [`np.unravel_index`](https://numpy.org/doc/stable/reference/generated/numpy.where.html) 函数，将平面索引转换为坐标索引；
比如在形状为[3,4]的坐标中，平面索引`6`代表拉平后的第六个元素，对应的坐标索引为第2行第3列，所以坐标索引为`(2,3)`。

参数：
- `indices:Var` 平面索引值
- `shape:Var` 坐标索引对应的形状
- `order:numpy兼容参数` 默认为None

返回：计算得到的坐标索引各轴的坐标值 

返回类型：`Var`数组

示例

```python
>>> np.unravel_index([22, 41, 37], (7,6))
[array([3, 6, 6], dtype=int32), array([4, 5, 1], dtype=int32)]
```
---
### `empty_like(prototype, dtype=None, order='K', subok=True, shape=None)`
作用等同与 `numpy` 中的 [`np.empty_like`](https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html) 函数，创建一个与指定Var相同的未初始化Var变量，是`expr.placeholder`的封装。

参数：
- `prototype:Var` 输出Var的形状和数据类型与该Var相同
- `dtype:dtype` 重新指定输出的数据类型
- `order:numpy兼容参数` 默认为None
- `subok:numpy兼容参数` 默认为None
- `shape:[int]` 重新指定输出的形状

返回：创建的空`Var` 

返回类型：`Var`

示例

```python
a = ([1,2,3], [4,5,6])
>>> x = np.empty_like(a)
```
---
### `arccos(x)`
作用等同与 `numpy` 中的 [`np.arccos`](https://numpy.org/doc/stable/reference/generated/numpy.arccos.html) 函数，按元素计算反余弦值。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.arccos([0.5, 1.0])
array([1.0471975, 0.       ], dtype=float32)
```
---
### `fabs(x)`
作用等同与 `numpy` 中的 [`np.fabs`](https://numpy.org/doc/stable/reference/generated/numpy.fabs.html) 函数，求输入的绝对值, 是`expr.fabs`的封装。

参数：
- `x:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.fabs([-1., 2.])
array([1., 2.], dtype=float32)
```
---
### `equal(a1, a2)`
作用等同与 `numpy` 中的 [`np.equal`](https://numpy.org/doc/stable/reference/generated/numpy.equal.html) 函数，判断一个变量是否等于另一个变量。


参数：
- `a1:Var` 输入的变量
- `a2:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.equal([2, 2], [1, 2])
array([0, 1], dtype=int32)
```
---
### `float_power(x1, x2)`
作用等同与 `numpy` 中的 [`np.float_power`](https://numpy.org/doc/stable/reference/generated/numpy.float_power.html) 函数，对输入的2个变量求指数, 是`expr.power`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.float_power(2., 3.)
array(8., dtype=float32)
```
---
### `arctanh(x)`
作用等同与 `numpy` 中的 [`np.arctanh`](https://numpy.org/doc/stable/reference/generated/numpy.arctanh.html) 函数，按元素计算双曲反正切值。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.arctanh([0.5, 0.8])
array([0.54930615, 1.0986123 ], dtype=float32)
```
---
### `arcsin(x)`
作用等同与 `numpy` 中的 [`np.arcsin`](https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html) 函数，按元素计算反正弦值。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.arcsin([0.5, 1.0])
array([0.5235988, 1.5707963], dtype=float32)
```
---
### `abs(x)`
作用等同与 `numpy` 中的 [`np.abs`](https://numpy.org/doc/stable/reference/generated/numpy.abs.html) 函数，求输入的绝对值, 是`expr.abs`的封装。

参数：
- `x:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.abs([-1, 2])
array([1, 2], dtype=int32)
```
---
### `var(a, axis=None, dtype=float32, out=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.var`](https://numpy.org/doc/stable/reference/generated/numpy.var.html) 函数，返回沿axis轴的方差。


参数：
- `a:Var` 将要统计的变量
- `axis:int` 统计的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留计数维度

返回：计数得到的变量 

返回类型：`float` of `Var` 

示例

```python
>>> a = np.arange(4).reshape((2,2))
>>> np.var(a)
1.25
>>> np.var(a, axis=0) 
array([1., 1.], dtype=float32)
```
---
### `multiply(x1, x2)`
作用等同与 `numpy` 中的 [`np.multiply`](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html) 函数，对输入的2个变量相乘, 是`expr.multiply`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.multiply(13, 17)
array(221, dtype=int32)
```
---
### `arcsinh(x)`
作用等同与 `numpy` 中的 [`np.arcsinh`](https://numpy.org/doc/stable/reference/generated/numpy.arcsinh.html) 函数，按元素计算双曲反正弦值。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.arcsinh([0.5, 1.0])
array([0.4812118, 0.8813736], dtype=float32)
```
---
### `array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0, like=None)`
作用等同与 `numpy` 中的 [`np.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html) 根据指定数据，类型创建一个Var。

参数：
- `object:ndarray` 输出变量的数据来源
- `dtype:dtype` 指定输出的类型; 默认为`object`的类型
- `order:numpy兼容参数` 默认为None
- `subok:numpy兼容参数` 默认为None

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> np.array([1, 2, 3])
array([1, 2, 3])
```
---
### `vdot(a, b)`
作用等同与 `numpy` 中的 [`np.vdot`](https://numpy.org/doc/stable/reference/generated/numpy.vdot.html) 函数，对输入的2个变量做点乘。
根据`a`和`b`的维度进行如下计算：
- 如果都是0维，等价于乘法
- 如果都是1维，等价于秋内积，即按位乘之后求和
- 如果都是2维，等价于矩阵乘法
- 如果`a`是N维，`b`是1维，则对最后一维度求内积
- 如果`a`是N维，`b`是M维(M>=2)，则对`a`最后一维和`b`的倒数第二维度求内积


参数：
- `a:Var` 参与计算的变量
- `b:Var` 参与计算的变量
- `out:numpy兼容参数` 默认为None

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.vdot(3, 4)
12
>>> np.vdot([[1, 0], [0, 1]], [[4, 1], [2, 2]])
array([[4, 1],
     [2, 2]], dtype=int32)
```
---
### `asfarray(a, dtype=np.float32)`
作用等同与 `numpy` 中的 [`np.asfarray`](https://numpy.org/doc/stable/reference/generated/numpy.asfarray.html) 根据指定数据，将输入数据转换为一个类型为float的Var。

参数：
- `a:ndarray` 输出变量的数据来源
- `dtype:dtype` 重新指定输出的类型; 默认为np.float32

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> np.asfarray([1, 2, 3])
array([1., 2., 3.])
```
---
### `remainder(x1, x2)`
作用等同与 `numpy` 中的 [`np.remainder`](https://numpy.org/doc/stable/reference/generated/numpy.remainder.html) 函数，对输入的2个变量求模, 是`expr.mod`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.remainder(13, 5)
array(3, dtype=int32)
```
---
### `amax(a, axis=None, out=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.amax`](https://numpy.org/doc/stable/reference/generated/numpy.amax.html) 函数，返回沿axis轴的最大值。


参数：
- `a:Var` 将要统计的变量
- `axis:int` 统计的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留计数维度

返回：计数得到的变量 

返回类型：`int` of `Var` 

示例

```python
>>> a = np.arange(4).reshape((2,2))
>>> np.amax(a) 
3
>>> np.amin(a, axis=0) 
array([2, 3], dtype=int32)
```
---
### `power(x1, x2)`
作用等同与 `numpy` 中的 [`np.power`](https://numpy.org/doc/stable/reference/generated/numpy.power.html) 函数，对输入的2个变量求指数, 是`expr.power`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.power(2., 3.)
array(8., dtype=float32)
```
---
### `nanmax(a, axis=None, out=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.nanmax`](https://numpy.org/doc/stable/reference/generated/numpy.nanmax.html) 函数，返回沿axis轴的最大值。


参数：
- `a:Var` 将要统计的变量
- `axis:int` 统计的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留计数维度

返回：计数得到的变量 

返回类型：`int` of `Var` 

示例

```python
>>> a = np.arange(4).reshape((2,2))
>>> np.nanmax(a) 
3
>>> np.nanmax(a, axis=0) 
array([2, 3], dtype=int32)
```
---
### `empty(shape, dtype=float32, order='C')`
作用等同与 `numpy` 中的 [`np.empty`](https://numpy.org/doc/stable/reference/generated/numpy.empty.html) 函数，创建一个指定形状未初始化的Var变量，是`expr.placeholder`的封装。

参数：
- `shape:tuple` 指定输出的形状
- `dtype:dtype` 指定输出的形状类型; 默认为`np.float32`
- `order:numpy兼容参数` 默认为None

返回：创建的空`Var` 

返回类型：`Var`

示例

```python
>>> x = np.empty([2, 2])
x.write([1.,2.,3.,4.])
```
---
### `arange([start, ]stop, [step, ], dtype=None)`
作用等同与 `numpy` 中的 [`np.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html) 根据指定数据，类型创建一个指定范围和步长的Var。

参数：
- `start:int|float` 输出变量的范围起点
- `stop:int|float` 输出变量的范围终点
- `step:int|float` 输出变量的范围步长
- `dtype:dtype` 指定输出的类型; 默认为`stop`的类型

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> np.arange(0, 5, 1)
array([0, 1, 2, 3, 4])
```
---
### `log2(x)`
作用等同与 `numpy` 中的 [`np.log2`](https://numpy.org/doc/stable/reference/generated/numpy.log2.html) 函数，按元素计算`log2(x)`。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.log2([1.,2.])
array([0., 1.], dtype=float32)
```
---
### `where(condition, x=None, y=None)`
作用等同与 `numpy` 中的 [`np.where`](https://numpy.org/doc/stable/reference/generated/numpy.where.html) 函数，根据给定条件返回两个输入的元素。

参数：
- `condition:Var` 条件变量
- `x:Var` 参与取值的变量
- `y:Var` 参与取值的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> a = np.arange(10)
>>> np.where(a < 5, a, a*10)
array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90], dtype=int32)
```
---
### `all(a, axis=None, out=None, keepdims=None)`
作用等同与 `numpy` 中的 [`np.all`](https://numpy.org/doc/stable/reference/generated/numpy.all.html) 函数，判断是否所有元素都为真(不为0)。


参数：
- `a:Var` 输入的变量
- `axis:list` 判断的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留指定轴的维度

返回：计算得到的变量 

返回类型：`Var`or`scalar`

示例

```python
>>> np.all([1, 0, 1])
False
>>> np.all([1, -1, 1])
True
```
---
### `round(x)`
作用等同与 `numpy` 中的 [`np.round`](https://numpy.org/doc/stable/reference/generated/numpy.round.html) 函数，按位进行四舍五入。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.round([1.2, 3.5, 4.7])
array([1., 4., 5.], dtype=float32)
```
---
### `divide(x1, x2)`
作用等同与 `numpy` 中的 [`np.divide`](https://numpy.org/doc/stable/reference/generated/numpy.divide.html) 函数，对输入的2个变量相除, 是`expr.divide`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.divide(13., 17.)
array(0.7647059, dtype=float32)
```
---
### `average(a, axis=None, weights=None, returned=False)`
作用等同与 `numpy` 中的 [`np.average`](https://numpy.org/doc/stable/reference/generated/numpy.average.html) 函数，返回沿axis轴的平均值。


参数：
- `a:Var` 将要统计的变量
- `axis:int` 统计的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留计数维度

返回：计数得到的变量 

返回类型：`float` of `Var` 

示例

```python
>>> a = np.arange(4).reshape((2,2))
>>> np.average(a)
1.5
>>> np.average(a, axis=0) 
array([1., 2.], dtype=float32)
```
---
### `trunc(x)`
作用等同与 `numpy` 中的 [`np.trunc`](https://numpy.org/doc/stable/reference/generated/numpy.trunc.html) 函数，按位进行四舍五入。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.trunc([1.2, 3.5, 4.7])
array([1., 4., 5.], dtype=float32)
```
---
### `nanprod(a, axis=None, out=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.nanprod`](https://numpy.org/doc/stable/reference/generated/numpy.nanprod.html) 函数，沿着指定维度，对数据求乘积。


参数：
- `x:Var` 输入的变量
- `axis:list` 计算的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留指定轴的维度

返回：计算得到的变量 

返回类型：`Var`or`scalar`

示例

```python
>>> np.nanprod([1,2,3,4])
24
```
---
### `std(a, axis=None, dtype=float32, out=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.std`](https://numpy.org/doc/stable/reference/generated/numpy.std.html) 函数，返回沿axis轴的标准差。


参数：
- `a:Var` 将要统计的变量
- `axis:int` 统计的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留计数维度

返回：计数得到的变量 

返回类型：`float` of `Var` 

示例

```python
>>> a = np.arange(4).reshape((2,2))
>>> np.std(a)
1.1180340051651
>>> np.std(a, axis=0) 
array([0.99999994, 0.99999994], dtype=float32)
```
---
### `bitwise_xor(x1, x2)`
作用等同与 `numpy` 中的 [`np.bitwise_xor`](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_xor.html) 函数，对输入的2个变量按位与, 是`expr.bitwise_xor`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.bitwise_xor(13, 17)
array(28, dtype=int32)
```
---
### `dot(a, b, out=None)`
作用等同与 `numpy` 中的 [`np.dot`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html) 函数，对输入的2个变量做点乘。
根据`a`和`b`的维度进行如下计算：
- 如果都是0维，等价于乘法
- 如果都是1维，等价于秋内积，即按位乘之后求和
- 如果都是2维，等价于矩阵乘法
- 如果`a`是N维，`b`是1维，则对最后一维度求内积
- 如果`a`是N维，`b`是M维(M>=2)，则对`a`最后一维和`b`的倒数第二维度求内积


参数：
- `a:Var` 参与计算的变量
- `b:Var` 参与计算的变量
- `out:numpy兼容参数` 默认为None

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.dot(3, 4)
12
>>> np.dot([[1, 0], [0, 1]], [[4, 1], [2, 2]])
array([[4, 1],
     [2, 2]], dtype=int32)
```
---
### `moveaxis(a, source, destination)`
作用等同与 `numpy` 中的 [`np.moveaxis`](https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html) 函数，移动`Var`的指定维度到指定位置。

参数：
- `a:Var` 将会移动该变量的维度
- `source:int` 需要移动的维度
- `destination:int` 移动的目标位置

返回：形状为newshape的`MNN.Var`

返回类型：`Var`

示例

```python
>>> a = np.ones((3, 4, 5))
>>> np.moveaxis(a, 0, -1).shape
(4, 5, 3)
```
---
### `flatnonzero(a)`
作用等同与 `numpy` 中的 [`np.flatnonzero`](https://numpy.org/doc/stable/reference/generated/numpy.flatnonzero.html) 函数，平铺之后输出非零元素的索引。

参数：
- `a:Var` 将要搜索的变量

返回：搜索得到的变量 

返回类型：`tuple` of `Var` 

示例

```python
>>> x = np.arange(6).reshape(2,3)
>>> np.flatnonzero(x>1)
array([2, 3, 4, 5], dtype=int32)
```
---
### `copysign(x1, x2)`
作用等同与 `numpy` 中的 [`np.copysign`](https://numpy.org/doc/stable/reference/generated/numpy.copysign.html) 函数，将第二个输入的符号作用到第一个输入上并返回。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.copysign(13, -1)
array(-13, dtype=int32)
```
---
### `atleast_3d(*arys)`
作用等同与 `numpy` 中的 [`np.atleast_2d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_3d.html) 函数，将输入`Var`的维度变为至少为3，如果输入为标量或1/2维变量则输出为3维变量，否则保持维度不变。

参数：
- `*arys:Var` 多个需要转换的变量

返回：转置后的 `MNN.Var`

返回类型：`Var`

示例

```python
>>> np.atleast_1d(3, [2,3])
[array([[[3]]], dtype=int32), array([[[2],[3]]], dtype=int32)]
```
---
### `sinc(x)`
作用等同与 `numpy` 中的 [`np.sinc`](https://numpy.org/doc/stable/reference/generated/numpy.sinc.html) 函数，对于输入`x`进行如下计算`sin(x*PI)/(x*PI)`。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.sinc([1.,2.])
array([-2.7827534e-08,  2.7827534e-08], dtype=float32)
```
---
### `modf(x)`
作用等同与 `numpy` 中的 [`np.modf`](https://numpy.org/doc/stable/reference/generated/numpy.modf.html) 函数, 按元素返回输入的整数部分和小数部分。

参数：
- `x:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`tuple` of `Var` 

示例

```python
>>> np.modf([2.1, -1.2])
(array([ 0.0999999 , -0.20000005], dtype=float32), array([ 2., -2.], dtype=float32))
```
---
### `nanmean(a, axis=None, dtype=float32, out=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.nanmean`](https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html) 函数，返回沿axis轴的平均值。


参数：
- `a:Var` 将要统计的变量
- `axis:int` 统计的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留计数维度

返回：计数得到的变量 

返回类型：`float` of `Var` 

示例

```python
>>> a = np.arange(4).reshape((2,2))
>>> np.nanmean(a)
1.5
>>> np.nanmean(a, axis=0) 
array([1., 2.], dtype=float32)
```
---
### `cosh(x)`
作用等同与 `numpy` 中的 [`np.cosh`](https://numpy.org/doc/stable/reference/generated/numpy.cosh.html) 函数，按元素计算双曲余弦值。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.cosh([1.,2.])
array([1.5430807, 3.7621956], dtype=float32)
```
---
### `log1p(x)`
作用等同与 `numpy` 中的 [`np.log1p`](https://numpy.org/doc/stable/reference/generated/numpy.log1p.html) 函数，按元素计算`log(x+1)`。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.log1p([1.,2.])
array([0.6931472, 1.0986123], dtype=float32)
```
---
### `maximum(x1, x2)`
作用等同与 `numpy` 中的 [`np.maximum`](https://numpy.org/doc/stable/reference/generated/numpy.maximum.html) 函数，输入2个变量中的最大值, 是`expr.maximum`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.maximum(13, 17)
array(17, dtype=int32)
```
---
### `ceil(x)`
作用等同与 `numpy` 中的 [`np.ceil`](https://numpy.org/doc/stable/reference/generated/numpy.ceil.html) 函数，按位对小数部分进位。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.ceil([1.2, 3.5, 4.7])
array([2., 4., 5.], dtype=float32)
```
---
### `log10(x)`
作用等同与 `numpy` 中的 [`np.log10`](https://numpy.org/doc/stable/reference/generated/numpy.log10.html) 函数，按元素计算`log10(x)`。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.log10([1.,2.])
array([0., 0.30102998], dtype=float32)
```
---
### `inner(a, b)`
作用等同与 `numpy` 中的 [`np.inner`](https://numpy.org/doc/stable/reference/generated/numpy.inner.html) 函数，对输入的2个变量做内积，即按位乘后求和。


参数：
- `a:Var` 参与计算的变量
- `b:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.inner([1,2,3], [0,1,0])
2
```
---
### `array_split(ary, indices_or_sections, axis=0)`
作用等同与 `numpy` 中的 [`np.array_split`](https://numpy.org/doc/stable/reference/generated/numpy.array_split.html) 函数，将一个`Var`拆分成为多个子变量。
该实现与`split`等价；与numpy的区别是当分块数目无法整除时不会保留余数部分。

参数：
- `ary:Var` 被拆分的Var
- `indices_or_sections:list` 拆分的块数或者块大小
- `axis:int` 拆分的维度

返回：拆分得到的`Var` 

返回类型：`Var` 数组

示例

```python
>>> x = np.arange(9.0)
>>> np.array_split(x, 3)
[array([0.,  1.,  2.]), array([3.,  4.,  5.]), array([6.,  7.,  8.])]
```
---
### `greater(a1, a2)`
作用等同与 `numpy` 中的 [`np.greater`](https://numpy.org/doc/stable/reference/generated/numpy.greater.html) 函数，判断一个变量是否大于另一个变量。


参数：
- `a1:Var` 输入的变量
- `a2:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.greater([2, 1], [1, 2])
array([1, 0], dtype=int32)
```
---
### `tile(A, reps)`
作用等同与 `numpy` 中的 [`np.tile`](https://numpy.org/doc/stable/reference/generated/numpy.tile.html) 函数，将输入变量复制指定次数构造一个新变量。

参数：
- `A:Var` 被复制的Var
- `reps:int` 复制的次数

返回：复制得到的`Var` 

返回类型：`Var` 

示例

```python
>>> a = np.array([0, 1, 2])
>>> np.tile(a, 2)
array([0, 1, 2, 0, 1, 2], dtype=int32)
```
---
### `argwhere(a)`
作用等同与 `numpy` 中的 [`np.argwhere`](https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html) 函数，输出非零元素的索引。

参数：
- `a:Var` 将要搜索的变量

返回：搜索得到的变量 

返回类型：`Var` 

示例

```python
>>> x = np.arange(6).reshape(2,3)
>>> np.argwhere(x>1)
array([[0, 2],
     [1, 0],
     [1, 1],
     [1, 2]], dtype=int32)
```
---
### `ones_like(a, dtype=None, order='K', subok=True, shape=None)`
作用等同与 `numpy` 中的 [`np.ones_like`](https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html) 函数，创建一个与指定Var的形状类型相同的Var，其全部的值为1, 为`full_like`的特例。

参数：
- `a:Var` 输出Var的形状和数据类型与该Var相同
- `dtype:dtype` 指定输出的形状类型; 默认为`np.float32`
- `order:numpy兼容参数` 默认为None
- `subok:numpy兼容参数` 默认为None
- `shape:[int]` 重新指定输出的形状

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> a = np.empty([2,2])
>>> np.ones_like(a)
array([[1., 1.],
     [1., 1.]])
```
---
### `bitwise_and(x1, x2)`
作用等同与 `numpy` 中的 [`np.bitwise_and`](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_and.html) 函数，对输入的2个变量按位与, 是`expr.bitwise_and`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.bitwise_and(13, 17)
array(1, dtype=int32)
```
---
### `lexsort(a, axis=-1)`
作用等同与 `numpy` 中的 [`np.lexsort`](https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html) 函数，对输入变量排序。

参数：
- `a:Var` 将要排序的变量
- `axis:int` 排序的维度

返回：排序后得到的变量 

返回类型：`Var` 

示例

```python
>>> a = np.array([[1,4],[3,1]])
>>> np.lexsort(a)  
array([[1, 4],
     [1, 3]], dtype=int32)
```
---
### `linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)`
作用等同与 `numpy` 中的 [`np.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html) 根据指定数据，类型创建一个指定范围（等差）和数目的Var。

参数：
- `start:int|float` 输出变量的范围起点
- `stop:int|float` 输出变量的范围终点
- `num:int` 输出数据总数
- `endpoint:bool` 目前仅支持endpoint=False
- `retstep:，返回值是否包含步长；如果包含则返回(Var` 返回值是否包含步长；如果包含则返回`(Var, step)`
- `dtype:dtype` 指定输出的类型; 默认为`stop`的类型
- `axis:numpy兼容参数` 默认为None

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> np.linspace(0, 10, 2, False, True)
(array([0, 5], dtype=int32), 5.0)
```
---
### `nansum(a, axis=None, out=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.nansum`](https://numpy.org/doc/stable/reference/generated/numpy.nansum.html) 函数，沿着指定维度，对数据求和。


参数：
- `x:Var` 输入的变量
- `axis:list` 计算的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留指定轴的维度

返回：计算得到的变量 

返回类型：`Var`or`scalar`

示例

```python
>>> np.nansum([1,2,3,4])
10
```
---
### `expm1(x)`
作用等同与 `numpy` 中的 [`np.expm1`](https://numpy.org/doc/stable/reference/generated/numpy.expm1.html) 函数，按元素计算`exp(x)-1`。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.expm1([1.,2.])
array([1.7182794, 6.388731 ], dtype=float32)
```
---
### `atleast_2d(*arys)`
作用等同与 `numpy` 中的 [`np.atleast_2d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html) 函数，将输入`Var`的维度变为至少为2，如果输入为标量或1维变量则输出为2维变量，否则保持维度不变。

参数：
- `*arys:Var` 多个需要转换的变量

返回：转置后的 `MNN.Var`

返回类型：`Var`

示例

```python
>>> np.atleast_1d(3, [2,3])
[array([[3]], dtype=int32), array([[2, 3]], dtype=int32)]
```
---
### `broadcast_to(array, shape)`
作用等同与 `numpy` 中的 [`np.broadcast_to`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html) 函数，将输入广播到指定形状。

参数：
- `array:Var` 需要广播的变量
- `shape:[int]` 广播的形状

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> a = np.array([1, 2, 3])
>>> np.broadcast_to(a, (3,3))
array([[1, 2, 3],
     [1, 2, 3],
     [1, 2, 3]], dtype=int32)
```
---
### `fmin(x1, x2)`
作用等同与 `numpy` 中的 [`np.fmin`](https://numpy.org/doc/stable/reference/generated/numpy.fmin.html) 函数，输入2个变量中的最小值, 是`expr.minimum`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.fmin(13., 17.)
array(13., dtype=float32)
```
---
### `min(a, axis=None, out=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.min`](https://numpy.org/doc/stable/reference/generated/numpy.min.html) 函数，返回沿axis轴的最小值。

参数：
- `a:Var` 将要统计的变量
- `axis:int` 统计的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留计数维度

返回：计数得到的变量 

返回类型：`int` of `Var` 

示例

```python
>>> a = np.arange(4).reshape((2,2))
>>> np.min(a) 
0
>>> np.min(a, axis=0) 
array([0, 1], dtype=int32)
```
---
### `geomspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)`
作用等同与 `numpy` 中的 [`np.geomspace`](https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html) 根据指定数据，类型创建一个指定范围(指数等差)和数目的Var。

参数：
- `start:int|float` 输出变量的范围起点
- `stop:int|float` 输出变量的范围终点
- `num:int` 输出数据总数
- `endpoint:bool` 是否包含终点元素，目前仅支持endpoint=False
- `retstep:(Var, step)` 返回值是否包含步长；如果包含则返回`(Var, step)`
- `dtype:dtype` 指定输出的类型; 默认为`stop`的类型
- `axis:numpy兼容参数` 默认为None

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> np.geomspace(1., 10., 4, False)
array([1.       , 1.7782794, 3.1622777, 5.6234136], dtype=float32)
```
---
### `add(x1, x2)`
作用等同与 `numpy` 中的 [`np.add`](https://numpy.org/doc/stable/reference/generated/numpy.add.html) 函数，对输入的2个变量相加, 是`expr.add`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.add(13, 17)
array(30, dtype=int32)
```
---
### `nonzero(a)`
作用等同与 `numpy` 中的 [`np.nonzero`](https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html) 函数，输出非零元素的索引。

参数：
- `a:Var` 将要搜索的变量

返回：搜索得到的变量 

返回类型：`tuple` of `Var` 

示例

```python
>>> x = np.arange(6).reshape(2,3)
>>> np.nonzero(x>1)
(array([0, 1, 1, 1], dtype=int32), array([2, 0, 1, 2], dtype=int32))
```
---
### `full(shape, fill_value, dtype=None, order='C')`
作用等同与 `numpy` 中的 [`np.full`](https://numpy.org/doc/stable/reference/generated/numpy.full.html) 函数，创建一个指定形状，类型的Var，其全部的值为指定数据。

参数：
- `shape:[int]` 指定输出的形状
- `fill_value:scalar` 指定输出的填充值
- `dtype:dtype` 指定输出的形状类型; 默认为`np.float32`
- `order:numpy兼容参数` 默认为None

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> np.full((2, 2), 10)
array([[10, 10],
     [10, 10]])
```
---
### `copy(a, order='K', subok=False)`
作用等同与 `numpy` 中的 [`np.copy`](https://numpy.org/doc/stable/reference/generated/numpy.copy.html) 函数，创建一个输入Var的拷贝。

参数：
- `a:Var` 输出Var的与该Var相同
- `order:numpy兼容参数` 默认为None
- `subok:numpy兼容参数` 默认为None

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> a = np.ones([2,2])
>>> np.copy(a)
array([[1., 1.],
     [1., 1.]])
```
---
### `subtract(x1, x2)`
作用等同与 `numpy` 中的 [`np.subtract`](https://numpy.org/doc/stable/reference/generated/numpy.subtract.html) 函数，对输入的2个变量相减, 是`expr.subtract`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.subtract(2., 3.)
array(-1., dtype=float32)
```
---
### `atleast_1d(*arys)`
作用等同与 `numpy` 中的 [`np.atleast_1d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_1d.html) 函数，将输入`Var`的维度变为至少为1，如果输入为标量则输出为1维变量，否则保持维度不变。

参数：
- `*arys:Var` 多个需要转换的变量

返回：转置后的 `MNN.Var`

返回类型：`Var`

示例

```python
>>> np.atleast_1d(3, [2,3])
[array([3], dtype=int32), array([2, 3], dtype=int32)]
```
---
### `nanstd(a, axis=None, dtype=float32, out=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.nanstd`](https://numpy.org/doc/stable/reference/generated/numpy.nanstd.html) 函数，返回沿axis轴的标准差。


参数：
- `a:Var` 将要统计的变量
- `axis:int` 统计的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留计数维度

返回：计数得到的变量 

返回类型：`float` of `Var` 

示例

```python
>>> a = np.arange(4).reshape((2,2))
>>> np.nanstd(a)
1.1180340051651
>>> np.nanstd(a, axis=0) 
array([0.99999994, 0.99999994], dtype=float32)
```
---
### `hsplit(ary, indices_or_sections)`
作用等同与 `numpy` 中的 [`np.vsplit`](https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html) 函数，沿着axis=0, 将一个`Var`拆分成为多个子变量。

参数：
- `ary:Var` 被拆分的Var
- `indices_or_sections:list` 拆分的块数或者块大小

返回：拆分得到的`Var` 

返回类型：`Var` 数组

示例

```python
>>> x = np.arange(16.0).reshape(2, 2, 4)
>>> np.vsplit(x, 2)
[array([[[0., 1., 2., 3.],
       [4., 5., 6., 7.]]], dtype=float32),
 array([[[ 8.,  9., 10., 11.],
       [12., 13., 14., 15.]]], dtype=float32)]
```
---
### `column_stack(tup)`
作用等同与 `numpy` 中的 [`np.column_stack`](https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html) 函数，把1维`Var`作为列stack成2位变量。

参数：
- `tup:Var序列` 被连接的变量

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> a = np.array((1,2,3))
>>> b = np.array((2,3,4))
>>> np.column_stack((a,b))
array([[1, 2],
     [2, 3],
     [3, 4]])
```
---
### `dstack(tup)`
作用等同与 `numpy` 中的 [`np.dstack`](https://numpy.org/doc/stable/reference/generated/numpy.dstack.html) 函数，深度顺序连接`Var`序列。

参数：
- `tup:Var序列` 被连接的变量

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> a = np.array([[1], [2], [3]])
>>> b = np.array([[4], [5], [6]])
>>> np.dstack((a, b))
array([[[1, 2]],
      [[2, 3]],
      [[3, 4]]])
```
---
### `reciprocal(x)`
作用等同与 `numpy` 中的 [`np.reciprocal`](https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html) 函数，按位取倒数。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.reciprocal([1.,2.])
array([1. , 0.5], dtype=float32)
```
---
### `positive(x)`
作用等同与 `numpy` 中的 [`np.positive`](https://numpy.org/doc/stable/reference/generated/numpy.positive.html) 函数，按位取正，相当于拷贝。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.positive([1.,-2.])
array([1., -2.], dtype=float32)
```
---
### `array_equal(a1, a2)`
作用等同与 `numpy` 中的 [`np.array_equiv`](https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html) 函数，判断2个变量是否完全相等。


参数：
- `a1:Var` 输入的变量
- `a2:Var` 输入的变量

返回：计算得到的变量 

返回类型：`bool`

示例

```python
>>> np.array_equiv([1, 2], [1, 2])
True
```
---
### `square(x)`
作用等同与 `numpy` 中的 [`np.square`](https://numpy.org/doc/stable/reference/generated/numpy.square.html) 函数，求输入的平方, 是`expr.square`的封装。

参数：
- `x:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.square(13)
array(160, dtype=int32)
```
---
### `sinh(x)`
作用等同与 `numpy` 中的 [`np.sinh`](https://numpy.org/doc/stable/reference/generated/numpy.sinh.html) 函数，按元素计算双曲正弦值。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.sinh([1.,2.])
array([1.1752012, 3.6268604], dtype=float32)
```
---
### `greater_equal(a1, a2)`
作用等同与 `numpy` 中的 [`np.greater_equal`](https://numpy.org/doc/stable/reference/generated/numpy.greater_equal.html) 函数，判断一个变量是否大于等于另一个变量。


参数：
- `a1:Var` 输入的变量
- `a2:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.greater_equal([2, 2], [1, 2])
array([1, 1], dtype=int32)
```
---
### `tanh(x)`
作用等同与 `numpy` 中的 [`np.tanh`](https://numpy.org/doc/stable/reference/generated/numpy.tanh.html) 函数，按元素计算双曲正切值。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.tanh([1.,2.])
array([0.7616205, 0.9640294], dtype=float32)
```
---
### `floor_divide(x1, x2)`
作用等同与 `numpy` 中的 [`np.floor_divide`](https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html) 函数，对输入的2个变量相除并舍去小数部分, 是`expr.floordiv`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.floor_divide(13., 17.)
array(0., dtype=float32)
```
---
### `matmul(a, b)`
作用等同与 `numpy` 中的 [`np.matmul`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) 函数，对输入的2个变量做矩阵乘。


参数：
- `a:Var` 参与计算的变量
- `b:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.matmul([[1, 0], [0, 1]], [[4, 1], [2, 2]])
array([[4, 1],
     [2, 2]], dtype=int32)
```
---
### `split(ary, indices_or_sections, axis=0)`
作用等同与 `numpy` 中的 [`np.split`](https://numpy.org/doc/stable/reference/generated/numpy.split.html) 函数，将一个`Var`拆分成为多个子变量。

参数：
- `ary:Var` 被拆分的Var
- `indices_or_sections:list` 拆分的块数或者块大小
- `axis:int` 拆分的维度

返回：拆分得到的`Var` 

返回类型：`Var` 数组

示例

```python
>>> x = np.arange(9.0)
>>> np.split(x, 3)
[array([0.,  1.,  2.]), array([3.,  4.,  5.]), array([6.,  7.,  8.])]
```
---
### `argsort(a, axis=-1, kind=None, order=None)`
作用等同与 `numpy` 中的 [`np.argsort`](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html) 函数，对输入变量排序,输出排序后的下标索引。

参数：
- `a:Var` 将要排序的变量
- `axis:int` 排序的维度
- `kind:numpy兼容参数` 默认为None
- `order:numpy兼容参数` 默认为None

返回：排序后得到的变量 

返回类型：`Var` 

示例

```python
>>> a = np.array([[1,4],[3,1]])
>>> np.argsort(a)  
array([[0, 1],
     [1, 0]], dtype=int32)
```
---
### `not_equal(a1, a2)`
作用等同与 `numpy` 中的 [`np.not_equal`](https://numpy.org/doc/stable/reference/generated/numpy.not_equal.html) 函数，判断一个变量是否不等于另一个变量。


参数：
- `a1:Var` 输入的变量
- `a2:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.not_equal([2, 2], [1, 2])
array([1, 0], dtype=int32)
```
---
### `broadcast_arrays(*args)`
作用等同与 `numpy` 中的 [`np.broadcast_arrays`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html) 函数，将输入的变量相互广播。

参数：
- `*args:Var` 需要广播的变量

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> x = np.array([[1,2,3]])
>>> y = np.array([[4],[5]])
>>> np.broadcast_arrays(x, y)
[array([[1, 2, 3],
        [1, 2, 3]], dtype=int32),
 array([[4, 4, 4],
        [5, 5, 5]], dtype=int32)]
```
---
### `squeeze(a, axis=None)`
作用等同与 `numpy` 中的 [`np.squeeze`](https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html) 函数，在指定位置移除一个维度，是expr.squeeze的封装。

参数：
- `x:Var` 需要移除维度的变量
- `axis:int`  移除维度的位置

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> a = np.array([[[0], [1], [2]]])
>>> b = np.squeeze(a, axis=0)
>>> a.shape
[1,3,1]
>>> b.shape
[3,1]
```
---
### `argmax(a, axis=None, out=None)`
作用等同与 `numpy` 中的 [`np.argmax`](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html) 函数，输出沿axis最大值的下标。

参数：
- `a:Var` 将要搜索的变量
- `axis:int` 搜索的维度
- `out:numpy兼容参数` 默认为None

返回：搜索得到的变量 

返回类型：`int` or `Var` 

示例

```python
>>> a = np.array([[1,4],[3,1]])
>>> np.argmax(a)  
1
>>> np.argmax(a, 0)  
array([1, 0], dtype=int32)
```
---
### `minimum(x1, x2)`
作用等同与 `numpy` 中的 [`np.minimum`](https://numpy.org/doc/stable/reference/generated/numpy.minimum.html) 函数，输入2个变量中的最小值, 是`expr.minimum`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.minimum(13, 17)
array(13, dtype=int32)
```
---
### `signbit(x)`
作用等同与 `numpy` 中的 [`np.signbit`](https://numpy.org/doc/stable/reference/generated/numpy.signbit.html) 函数，输入是否有负符号（小于0）。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.signbit([1.,-2.])
array([0, 1], dtype=int32)
```
---
### `around(x)`
作用等同与 `numpy` 中的 [`np.around`](https://numpy.org/doc/stable/reference/generated/numpy.around.html) 函数，按位进行四舍五入。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.around([1.2, 3.5, 4.7])
array([1., 4., 5.], dtype=float32)
```
---
### `meshgrid(*xi, copy=True, sparse=False, indexing='xy')`
作用等同与 `numpy` 中的 [`np.meshgrid`](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html) 根据坐标值返回坐标矩阵。

参数：
- `*xi:Var` 一维坐标值
- `copy:bool` 是否拷贝所有的值到输出
- `sparse:bool` 是否返回全部坐标矩阵
- `indexing:{‘xy’, ‘ij’}`  ‘ij’}`，'xy'会翻转坐标轴

返回：创建的`Var` 

返回类型：`Var`

示例

```python
nx, ny = (3, 2)
>>> x = np.linspace(0., 1., nx, False)
>>> y = np.linspace(0., 1., ny, False)
>>> xv, yv = np.meshgrid(x, y)
xv
array([[0.        , 0.33333334, 0.6666667 ],
     [0.        , 0.33333334, 0.6666667 ]], dtype=float32)
yv
array([[0. , 0. , 0. ],
     [0.5, 0.5, 0.5]], dtype=float32)
```
---
### `nanmin(a, axis=None, out=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.nanmin`](https://numpy.org/doc/stable/reference/generated/numpy.nanmin.html) 函数，返回沿axis轴的最小值。

参数：
- `a:Var` 将要统计的变量
- `axis:int` 统计的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留计数维度

返回：计数得到的变量 

返回类型：`int` of `Var` 

示例

```python
>>> a = np.arange(4).reshape((2,2))
>>> np.nanmin(a) 
0
>>> np.nanmin(a, axis=0) 
array([0, 1], dtype=int32)
```
---
### `amin(a, axis=None, out=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.amin`](https://numpy.org/doc/stable/reference/generated/numpy.amin.html) 函数，返回沿axis轴的最小值。

参数：
- `a:Var` 将要统计的变量
- `axis:int` 统计的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留计数维度

返回：计数得到的变量 

返回类型：`int` of `Var` 

示例

```python
>>> a = np.arange(4).reshape((2,2))
>>> np.amin(a) 
0
>>> np.amin(a, axis=0) 
array([0, 1], dtype=int32)
```
---
### `sqrt(x)`
作用等同与 `numpy` 中的 [`np.sin`](https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html) 函数，按元素计算平方根。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.sqrt([1.,2.])
array([0.99999994, 1.4142134 ], dtype=float32)
```
---
### `sum(a, axis=None, out=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.sum`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) 函数，沿着指定维度，对数据求和。


参数：
- `x:Var` 输入的变量
- `axis:[int]` 计算的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留指定轴的维度

返回：计算得到的变量 

返回类型：`Var`or`scalar`

示例

```python
>>> np.sum([1,2,3,4])
10
```
---
### `nanargmax(a, axis=None, out=None)`
作用等同与 `numpy` 中的 [`np.nanargmax`](https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html) 函数，输出沿axis最大值的下标。

参数：
- `a:Var` 将要搜索的变量
- `axis:int` 搜索的维度
- `out:numpy兼容参数` 默认为None

返回：搜索得到的变量 

返回类型：`int` or `Var` 

示例

```python
>>> a = np.array([[1,4],[3,1]])
>>> np.nanargmax(a)  
1
>>> np.nanargmax(a, 0)  
array([1, 0], dtype=int32)
```
---
### `log(x)`
作用等同与 `numpy` 中的 [`np.log`](https://numpy.org/doc/stable/reference/generated/numpy.log.html) 函数，按元素计算`log(x)`。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.log([1.,2.])
array([0. , 0.6931472], dtype=float32)
```
---
### `floor(x)`
作用等同与 `numpy` 中的 [`np.floor`](https://numpy.org/doc/stable/reference/generated/numpy.floor.html) 函数，按位舍去小数部分。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.floor([1.2, 3.5, 4.7])
array([1., 3., 4.], dtype=float32)
```
---
### `hypot(x1, x2)`
作用等同与 `numpy` 中的 [`np.hypot`](https://numpy.org/doc/stable/reference/generated/numpy.hypot.html) 函数，输入2个变量`x1`和`x2`执行计算`sqrt(x1*x1+x2*x2)`。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.hypot(13., 17.)
array(21.400934, dtype=float32)
```
---
### `vstack(tup)`
作用等同与 `numpy` 中的 [`np.vstack`](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html) 函数，垂直顺序连接`Var`序列。

参数：
- `tup:Var序列` 被连接的变量

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> a = np.array([[1], [2], [3]])
>>> b = np.array([[4], [5], [6]])
>>> np.vstack((a, b))
array([[1],
       [2],
       [3],
       [4],
       [5],
       [6]])
```
---
### `bitwise_or(x1, x2)`
作用等同与 `numpy` 中的 [`np.bitwise_or`](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_or.html) 函数，对输入的2个变量按位或, 是`expr.bitwise_or`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.bitwise_or(13, 17)
array(29, dtype=int32)
```
---
### `dsplit(ary, indices_or_sections)`
作用等同与 `numpy` 中的 [`np.dsplit`](https://numpy.org/doc/stable/reference/generated/numpy.dsplit.html) 函数，沿着axis=2, 将一个`Var`拆分成为多个子变量。

参数：
- `ary:Var` 被拆分的Var
- `indices_or_sections:list` 拆分的块数或者块大小

返回：拆分得到的`Var` 

返回类型：`Var` 数组

示例

```python
>>> x = np.arange(16.0).reshape(2, 2, 4)
>>> np.dsplit(x, 2)
[array([[[ 0.,  1.],
         [ 4.,  5.]],
        [[ 8.,  9.],
         [12., 13.]]], dtype=float32),
 array([[[ 2.,  3.],
         [ 6.,  7.]],
        [[10., 11.],
         [14., 15.]]], dtype=float32)]
```
---
### `full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None)`
作用等同与 `numpy` 中的 [`np.full_like`](https://numpy.org/doc/stable/reference/generated/numpy.full_like.html) 函数，创建一个与指定Var的形状类型相同的Var，其全部的值为指定数据。

参数：
- `a:Var` 输出Var的形状和数据类型与该Var相同
- `fill_value:scalar` 指定输出的填充值
- `dtype:dtype` 指定输出的形状类型; 默认为`np.float32`
- `order:numpy兼容参数` 默认为None
- `subok:numpy兼容参数` 默认为None
- `shape:[int]` 重新指定输出的形状

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> a = np.empty([2,2])
>>> np.full_like(a, 10)
array([[10, 10],
       [10, 10]])
```
---
### `identity(n, dtype=float32)`
作用等同与 `numpy` 中的 [`np.identity`](https://numpy.org/doc/stable/reference/generated/numpy.identity.html) 函数，创建一个二维Var, 对角线的值为1，是eye的特殊情况。

参数：
- `n:int` 指定输出的行和列
- `dtype:dtype` 指定输出的形状类型; 默认为`np.float32`

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> np.identity(3)
array([[1.,  0.,  0.],
     [0.,  1.,  0.],
     [0.,  0.,  1.]])
```
---
### `eye(N, M=None, k=0, dtype=float32, order='C')`
作用等同与 `numpy` 中的 [`np.eye`](https://numpy.org/doc/stable/reference/generated/numpy.eye.html) 函数，创建一个二维Var, 对角线的值为1。

参数：
- `N:int` 指定输出的行
- `M:int` 指定输出的列
- `k:int` 指定输出对角线位置
- `dtype:dtype` 指定输出的形状类型; 默认为`np.float32`
- `order:numpy兼容参数` 默认为None

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> np.eye(3, k=1)
array([[0.,  1.,  0.],
     [0.,  0.,  1.],
     [0.,  0.,  0.]])
```
---
### `absolute(x)`
作用等同与 `numpy` 中的 [`np.absolute`](https://numpy.org/doc/stable/reference/generated/numpy.absolute.html) 函数，求输入的绝对值, 是`expr.abs`的封装。

参数：
- `x:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.absolute([-1, 2])
array([1, 2], dtype=int32)
```
---
### `less(a1, a2)`
作用等同与 `numpy` 中的 [`np.less`](https://numpy.org/doc/stable/reference/generated/numpy.less.html) 函数，判断一个变量是否小于另一个变量。


参数：
- `a1:Var` 输入的变量
- `a2:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.less([2, 2], [1, 2])
array([0, 0], dtype=int32)
```
---
### `repmat(a, m, n)`
作用等同与 `numpy` 中的 [`np.repmat`](https://numpy.org/doc/stable/reference/generated/numpy.repmat.html) 根据指定数据，将0维数据复制成为2维(MxN)的Var。

参数：
- `a:ndarray` 被复制的数据
- `m:int` 复制的行数
- `n:int` 复制的列数

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> a = np.arange(3)
>>> np.repmat(a, 2, 3)
array([[0, 1, 2, 0, 1, 2, 0, 1, 2],
     [0, 1, 2, 0, 1, 2, 0, 1, 2]], dtype=int32)
```
---
### `arctan(x)`
作用等同与 `numpy` 中的 [`np.arctan`](https://numpy.org/doc/stable/reference/generated/numpy.arctan.html) 函数，按元素计算反正切值。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.arctan([0.5, 1.0])
array([0.4636476, 0.7853982], dtype=float32)
```
---
### `stack(arrays, axis=0, out=None)`
作用等同与 `numpy` 中的 [`np.stack`](https://numpy.org/doc/stable/reference/generated/numpy.stack.html) 函数，沿指定轴连接输入的`Var`连接轴会创建一个新的维度。

参数：
- `args:Var序列` 被连接的变量
- `axis:int` 指定连接的轴
- `out:numpy兼容参数` 默认为None

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> a = np.array([1, 2, 3])
>>> b = np.array([4, 5, 6])
>>> np.stack((a, b))
array([[1, 2, 3],
     [4, 5, 6]])
```
---
### `rint(x)`
作用等同与 `numpy` 中的 [`np.rint`](https://numpy.org/doc/stable/reference/generated/numpy.rint.html) 函数，按位进行四舍五入。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.rint([1.2, 3.5, 4.7])
array([1., 4., 5.], dtype=float32)
```
---
### `sort(a, axis=-1, kind=None, order=None)`
作用等同与 `numpy` 中的 [`np.sort`](https://numpy.org/doc/stable/reference/generated/numpy.sort.html) 函数，对输入变量排序。

参数：
- `a:Var` 将要排序的变量
- `axis:int` 排序的维度
- `kind:numpy兼容参数` 默认为None
- `order:numpy兼容参数` 默认为None

返回：排序后得到的变量 

返回类型：`Var` 

示例

```python
>>> a = np.array([[1,4],[3,1]])
>>> np.sort(a)  
array([[1, 4],
       [1, 3]], dtype=int32)
```
---
### `arctan2(x1, x2)`
作用等同与 `numpy` 中的 [`np.arctan2`](https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html) 函数，按元素计算输入比值的反正切值。


参数：
- `x1:Var` 输入的变量
- `x2:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.arctan2(2., 3.)
array(0.5880026, dtype=float32)
```
---
### `logaddexp(x1, x2)`
作用等同与 `numpy` 中的 [`np.logaddexp`](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html) 函数，输入2个变量`x1`和`x2`执行计算`log(exp(x1) + exp(x2))`。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.logaddexp(2, 3)
array(3.3132498, dtype=float32)
```
---
### `swapaxes(a, axis1, axis2)`
作用等同与 `numpy` 中的 [`np.swapaxes`](https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html) 函数，交换`Var`的两个指定维度。

参数：
- `a:Var` 将会交换该变量的维度
- `axis1:int` 交换的维度1
- `axis2:int` 交换的维度2

返回：交换维度后的 `MNN.Var`

返回类型：`Var`

示例

```python
>>> a = np.ones((3, 4, 5))
>>> np.swapaxes(a, 0, 2).shape
[5, 4, 3]
```
---
### `less_equal(a1, a2)`
作用等同与 `numpy` 中的 [`np.less_equal`](https://numpy.org/doc/stable/reference/generated/numpy.less_equal.html) 函数，判断一个变量是否小于等于另一个变量。


参数：
- `a1:Var` 输入的变量
- `a2:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.less_equal([2, 2], [1, 2])
array([0, 1], dtype=int32)
```
---
### `msort(a)`
作用等同与 `numpy` 中的 [`np.msort`](https://numpy.org/doc/stable/reference/generated/numpy.msort.html) 函数，沿axis=0对输入变量排序。

参数：
- `a:Var` 将要排序的变量

返回：排序后得到的变量 

返回类型：`Var` 

示例

```python
>>> a = np.array([[1,4],[3,1]])
>>> np.msort(a)  
array([[1, 1],
       [3, 4]], dtype=int32)
```
---
### `logspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)`
作用等同与 `numpy` 中的 [`np.logspace`](https://numpy.org/doc/stable/reference/generated/numpy.logspace.html) 根据指定数据，类型创建一个指定范围（对数等差）和数目的Var。

参数：
- `start:int|float` 输出变量的范围起点
- `stop:int|float` 输出变量的范围终点
- `num:int` 输出数据总数
- `endpoint:bool` 是否包含终点元素，目前仅支持endpoint=False
- `retstep:(Var, step)` 返回值是否包含步长；如果包含则返回`(Var, step)`
- `dtype:dtype` 指定输出的类型; 默认为`stop`的类型
- `axis:numpy兼容参数` 默认为None

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> np.logspace(0., 10., 5, False)
array([1.e+00, 1.e+02, 1.e+04, 1.e+06, 1.e+08], dtype=float32)
```
---
### `array_equal(a1, a2, equal_nan=False)`
作用等同与 `numpy` 中的 [`np.array_equal`](https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html) 函数，判断2个变量是否完全相等。


参数：
- `a1:Var` 输入的变量
- `a2:Var` 输入的变量
- `equal_nan:numpy兼容参数` 默认为None

返回：计算得到的变量 

返回类型：`bool`

示例

```python
>>> np.array_equal([1, 2], [1, 2])
True
```
---
### `nanvar(a, axis=None, dtype=float32, out=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.nanvar`](https://numpy.org/doc/stable/reference/generated/numpy.nanvar.html) 函数，返回沿axis轴的方差。


参数：
- `a:Var` 将要统计的变量
- `axis:int` 统计的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留计数维度

返回：计数得到的变量 

返回类型：`float` of `Var` 

示例

```python
>>> a = np.arange(4).reshape((2,2))
>>> np.nanvar(a)
1.25
>>> np.nanvar(a, axis=0) 
array([1., 1.], dtype=float32)
```
---
### `row_stack(tup)`
作用等同与 `numpy` 中的 [`np.row_stack`](https://numpy.org/doc/stable/reference/generated/numpy.row_stack.html) 函数，垂直顺序连接`Var`序列，与`vstack`相同。

参数：
- `tup:Var序列` 被连接的变量

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> a = np.array([[1], [2], [3]])
>>> b = np.array([[4], [5], [6]])
>>> np.row_stack((a, b))
array([[1],
       [2],
       [3],
       [4],
       [5],
       [6]])
```
---
### `negative(x)`
作用等同与 `numpy` 中的 [`np.negative`](https://numpy.org/doc/stable/reference/generated/numpy.negative.html) 函数，按位取负。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.negative([1.,-2.])
array([-1., 2.], dtype=float32)
```
---
### `tan(x)`
作用等同与 `numpy` 中的 [`np.tan`](https://numpy.org/doc/stable/reference/generated/numpy.tan.html) 函数，按元素计算正切值。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.tan([1.,2.])
array([ 1.5574077, -2.1850398], dtype=float32)
```
---
### `any(a, axis=None, out=None, keepdims=None)`
作用等同与 `numpy` 中的 [`np.any`](https://numpy.org/doc/stable/reference/generated/numpy.any.html) 函数，判断是否所有元素存在真(不为0)。


参数：
- `a:Var` 输入的变量
- `axis:list` 判断的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留指定轴的维度

返回：计算得到的变量 

返回类型：`Var`or`scalar`

示例

```python
>>> np.any([1, 0, 1])
True
>>> np.all([0, 0, 0])
False
```
---
### `arccosh(x)`
作用等同与 `numpy` 中的 [`np.arccosh`](https://numpy.org/doc/stable/reference/generated/numpy.arccosh.html) 函数，按元素计算双曲反余弦值。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.arccosh([2.0, 3.0])
array([1.316958 , 1.7627472], dtype=float32)
```
---
### `exp(x)`
作用等同与 `numpy` 中的 [`np.exp`](https://numpy.org/doc/stable/reference/generated/numpy.exp.html) 函数，按元素计算`exp(x)`。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.exp([1.,2.])
array([2.7182794, 7.388731 ], dtype=float32)
```
---
### `ldexp(x1, x2)`
作用等同与 `numpy` 中的 [`np.ldexp`](https://numpy.org/doc/stable/reference/generated/numpy.ldexp.html) 函数，输入2个变量`x1`和`x2`执行计算`x1 * exp2(x2)`。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.ldexp(2., 3.)
array(16., dtype=float32)
```
---
### `prod(a, axis=None, out=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.prod`](https://numpy.org/doc/stable/reference/generated/numpy.prod.html) 函数，沿着指定维度，对数据求乘积。


参数：
- `x:Var` 输入的变量
- `axis:list` 计算的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留指定轴的维度

返回：计算得到的变量 

返回类型：`Var`or`scalar`

示例

```python
>>> np.prod([1,2,3,4])
24
```
---
### `ravel(a, order='C')`
作用等同与 `numpy` 中的 [`np.ravel`](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html) 函数，改变输入`Var`的形状为一维。

参数：
- `a:Var` 将会改变该变量的形状
- `order:numpy兼容参数` 默认为None

返回：形状为1维的`MNN.Var`

返回类型：`Var`

示例

```python
>>> a = np.ones([[2],[2]])
>>> np.ravel(a)
array([1, 1, 1, 1], dtype=int32)
```
---
### `true_divide(x1, x2)`
作用等同与 `numpy` 中的 [`np.true_divide`](https://numpy.org/doc/stable/reference/generated/numpy.true_divide.html) 函数，对输入的2个变量相除, 是`expr.divide`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.true_divide(13., 17.)
array(0.7647059, dtype=float32)
```
---
### `exp2(x)`
作用等同与 `numpy` 中的 [`np.exp2`](https://numpy.org/doc/stable/reference/generated/numpy.exp2.html) 函数，按元素计算`2**x`。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.exp2([1.,2.])
array([2., 4.], dtype=float32)
```
---
### `fmax(x1, x2)`
作用等同与 `numpy` 中的 [`np.fmax`](https://numpy.org/doc/stable/reference/generated/numpy.fmax.html) 函数，输入2个变量中的最大值, 是`expr.maximum`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.fmax(13., 17.)
array(17., dtype=float32)
```
---
### `fmod(x1, x2)`
作用等同与 `numpy` 中的 [`np.fmod`](https://numpy.org/doc/stable/reference/generated/numpy.fmod.html) 函数，对输入的2个变量求模, 是`expr.mod`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.fmod(13., 5.)
array(3., dtype=float32)
```
---
### `mean(a, axis=None, dtype=float32, out=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.mean`](https://numpy.org/doc/stable/reference/generated/numpy.mean.html) 函数，返回沿axis轴的平均值。


参数：
- `a:Var` 将要统计的变量
- `axis:int` 统计的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留计数维度

返回：计数得到的变量 

返回类型：`float` of `Var` 

示例

```python
>>> a = np.arange(4).reshape((2,2))
>>> np.mean(a)
1.5
>>> np.mean(a, axis=0) 
array([1., 2.], dtype=float32)
```
---
### `max(a, axis=None, out=None, keepdims=False)`
作用等同与 `numpy` 中的 [`np.max`](https://numpy.org/doc/stable/reference/generated/numpy.max.html) 函数，返回沿axis轴的最大值。


参数：
- `a:Var` 将要统计的变量
- `axis:int` 统计的轴
- `out:numpy兼容参数` 默认为None
- `keepdims:bool` 是否保留计数维度

返回：计数得到的变量 

返回类型：`int` of `Var` 

示例

```python
>>> a = np.arange(4).reshape((2,2))
>>> np.max(a) 
3
>>> np.max(a, axis=0) 
array([2, 3], dtype=int32)
```
---
### `mod(x1, x2)`
作用等同与 `numpy` 中的 [`np.mod`](https://numpy.org/doc/stable/reference/generated/numpy.mod.html) 函数，对输入的2个变量求模, 是`expr.mod`的封装。

参数：
- `x1:Var` 参与计算的变量
- `x2:Var` 参与计算的变量

返回：计算得到的变量 

返回类型：`Var` 

示例

```python
>>> np.mod(13, 5)
array(3, dtype=int32)
```
---
### `asmatrix(a, dtype=None)`
作用等同与 `numpy` 中的 [`np.asmatrix`](https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html) 根据指定数据，将输入数据转换为ndim=2的Var。

参数：
- `a:ndarray` 输出变量的数据来源
- `dtype:dtype` 重新指定输出的类型; 默认为a的数据类型

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> np.asmatrix([1, 2, 3, 4])
array([[1, 2, 3]])
```
---
### `fix(x)`
作用等同与 `numpy` 中的 [`np.fix`](https://numpy.org/doc/stable/reference/generated/numpy.fix.html) 函数，按位进行四舍五入。


参数：
- `x:Var` 输入的变量

返回：计算得到的变量 

返回类型：`Var`

示例

```python
>>> np.fix([1.2, 3.5, 4.7])
array([1., 4., 5.], dtype=float32)
```
---
### `hstack(tup)`
作用等同与 `numpy` 中的 [`np.hstack`](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html) 函数，水平顺序连接`Var`序列。

参数：
- `tup:Var序列` 被连接的变量

返回：创建的`Var` 

返回类型：`Var`

示例

```python
>>> a = np.array([[1], [2], [3]])
>>> b = np.array([[4], [5], [6]])
>>> np.hstack((a, b))
array([[1, 4],
     [2, 5],
     [3, 6]])
```
---
### `transpose(a, axes=None)`
作用等同与 `numpy` 中的 [`np.transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html) 函数，按照指定维度转置`Var`。

参数：
- `a:Var` 将转置该变量的维度
- `axes:tuple|list` 转置的维度顺序

返回：转置后的 `MNN.Var`

返回类型：`Var`

示例

```python
>>> a = np.ones((3, 4, 5))
>>> np.transpose(a).shape
[5, 4, 3]
```