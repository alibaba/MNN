## expr.Var

```python
class Var
```
Var是MNN V3及表达式接口中的基础数据结构，是对数据和表达式的封装类型。Var存储了计算表达式，数据以及数据类型，形状等诸多信息，用户可以通过Var本身的函数获取这些信息。

在numpy和cv接口中，Var的作用等价于`numpy.ndarray`。

---
### `expr.dtype`
描述Var的数据类型
- 类型：`Enum`
- 枚举值：
  - `double`
  - `float`
  - `int`
  - `int64`
  - `uint8`

---
### `expr.data_format`
描述Var的数据排布格式
- 类型：`Enum`
- 枚举值：
  - `NCHW`
  - `NC4HW4`
  - `NHWC`

---
### `Var()`
使用`expr.const`或`MNN.numpy.array`创建Var

*不要使用Var()来创建Var的变量，应该使用expr/numpy中提供的函数来创建Var*

---
### `valid`

该Var是否有效，如果无效，则不能使用, *作为表达式输入异常或无效的表达式的输出，Var都是无效的*

返回：Var的数据形状

返回类型：`[int]`

---
### `shape`

获取Var的形状

返回：Var的数据形状

返回类型：`[int]`

---
### `data_format`

获取Var的数据排布格式

属性类型：只读

类型：`expr.data_format`

---
### `dtype`

获取Var的数据类型

属性类型：只读

类型：`expr.dtype`

---
### `size`

获取Var的元素个数

属性类型：只读

类型：`int`

---
### `name`

获取Var的名称

属性类型：只读

类型：`str`

---
### `ndim`

获取Var的维度数量

属性类型：只读

类型：`int`

---
### `ptr`

获取Var的数据指针

属性类型：只读

类型：`PyCapsule`

---
###  `op_type`

获取Var对应表达式的Op类型

属性类型：只读

类型：`str`

---
###  `fix_as_placeholder()`

将Var作为Placeholder，执行该函数后可以对该变量执行写操作

*注意：表达式中使用，numpy数值计算时勿使用*

参数：
- `None`

返回：`None`

返回类型：`None`

---
###  `fix_as_const()`

将Var作为Const，执行该操作会立即计算出该变量的值并设置为常量

*注意：表达式中使用，numpy数值计算时勿使用*

参数：
- `None`

返回：`None`

返回类型：`None`

---
###  `fix_as_trainable()`

将Var作为Trainable，执行该操作会将该变量设置为可训练变量

*注意：表达式中使用，numpy数值计算时勿使用*

参数：
- `None`

返回：`None`

返回类型：`None`

---
###  `close()`

将Var的输入设置为空

*注意：表达式中使用，numpy数值计算时勿使用*

参数：
- `None`

返回：`None`

返回类型：`None`

---
###  `copy_from(src)`

将src设置为Var的输入

*注意：表达式中使用，numpy数值计算时勿使用*

参数：
- `src:Var` 源变量

返回：`None`

返回类型：`None`

---
###  `set_inputs(inputs)`

将inputs设置为Var的输入

*注意：表达式中使用，numpy数值计算时勿使用*

参数：
- `inputs:[Var]` 源变量

返回：`None`

返回类型：`None`

---
###  `replace(src)`

使用src替换掉该Var

*注意：表达式中使用，numpy数值计算时勿使用*

参数：
- `src:Var` 替换变量

返回：`None`

返回类型：`None`

---
###  `reorder(format)`

将Var的数据排布格式设置为format

*注意：表达式中使用，numpy数值计算时勿使用*

参数：
- `format:expr.data_format` 数据排布格式

返回：`None`

返回类型：`None`

---
###  `resize(shape)`

将Var的形状设置为shape

*注意：表达式中使用，numpy数值计算时勿使用*

参数：
- `shape:[int]` 形状

返回：`None`

返回类型：`None`

---
###  `read()`

读取Var的数据，返回numpy.ndarray数据

*注意：该API仅在PYMNN_NUMPY_USABLE打开的情况下有效，移动端默认关闭*

参数：
- `None`

返回：Var数据的numpy形式

返回类型：`numpy.ndarray`

---
###  `read_as_tuple()`

读取Var的数据，返回tuple数据

参数：
- `None`

返回：Var数据的tuple形式

返回类型：`tuple`

---
###  `write(data)`

将data中的数据写入Var，data可以是numpy.ndarray或tuple类型

*注意：该函数在使用`numpy`时经常被用作`ndarray->Var`的转换函数，在使用`MNN.numpy`时不需要使用该函数*

参数：
- `data:tuple|ndarray` 待写入数据

返回：`None`

返回类型：`None`

---
###  `all(axis)`

Var中指定轴的数据是否全不为0，相当于: `for x in data: res &= x` 

*要求Var的数据类型为int32*

参数：
- `axis:[int]` 指定的轴，默认为[-1]

返回：`True`全不为0，`False`包含0

返回类型：`bool`

---
###  `any(axis)`

Var中指定轴的数据是否至少有一个数据不为0，相当于: `for x in data: res |= x` 

*要求Var的数据类型为int32*

参数：
- `axis:[int]` 指定的轴，默认为[-1]

返回：`True`至少有一个数据不为0，`False`全部为0

返回类型：`bool`

---
###  `argmax(axis)`

返回Var中指定轴的最大值的索引

参数：
- `axis:[int]` 指定的轴，默认为[-1]

返回：最大值的索引

返回类型：`int`

---
###  `argmin(axis)`

返回Var中指定轴的最小值的索引

参数：
- `axis:[int]` 指定的轴，默认为[-1]

返回：最小值的索引

返回类型：`int`

---
###  `sort(axis)`

将Var按照axis的方向排序

参数：
- `axis:int` 排序的轴，默认为[-1]

返回：排序后的Var

返回类型：`Var`

---
###  `argsort(axis)`

返回排序后各元素对应的位置

参数：
- `axis:[int]` 排序的轴，默认为[-1]

返回：元素的对应顺序

返回类型：`Var`

---
###  `astype(type)`

将Var的数据类型设置为type

参数：
- `type:expr.dtype` 数据类型

返回：数据类型为type的变量

返回类型：`Var`

---
###  `copy()`

返回一个拷贝的Var

参数：
- `None`

返回：一个拷贝的Var

返回类型：`Var`

---
###  `dot(b)`

返回Var与b的点积

参数：
- `b:Var` 另一个变量

返回：Var与b的点积

返回类型：`Var`

---
###  `fill(value)`

将Var的数据全部设置为value

参数：
- `value:scalar` 填充值

返回：全部设置为value的变量

返回类型：`Var`

---
###  `flatten()`

将Var的数据展平，相当于 `reshape(-1)`

参数：
- `None`

返回：展平的变量

返回类型：`Var`

---
###  `max(axis)`

返回Var指定轴的最大值

参数：
- `axis:[int]` 指定的轴，默认为[-1]

返回：最大值

返回类型：`float`

---
###  `mean(axis)`

返回Var指定轴的均值

参数：
- `axis:[int]` 指定的轴，默认为[-1]

返回：均值

返回类型：`float`

---
###  `min()`

返回Var中最小值

参数：
- `None`

返回：最小值

返回类型：`float`

---
###  `nonzero()`

返回Var中不为0的元素的坐标

参数：
- `None`

返回：不为0的元素的坐标

返回类型：`(Var,)`

---
###  `prod()`

返回Var的乘积

参数：
- `None`

返回：乘积

返回类型：`float`

---
###  `ptp()`

返回Var的最大值与最小值的差

参数：
- `None`

返回：最大值与最小值的差

返回类型：`float`

---
###  `ravel()`

返回Var的数据展平，相当于 `reshape(-1)`

参数：
- `None`

返回：展平的变量

返回类型：`Var`

---
###  `repeat(num)`

将Var重复num次

参数：
- `num:int` 重复次数

返回：重复num次的变量

返回类型：`Var`

---
###  `reshape(shape)`

将Var的数据reshape为shape

参数：
- `shape:[int]` 新的形状

返回：形状为shape的变量

返回类型：`Var`

---
###  `squeeze(axis)`

将Var中指定轴且维度为1的维度移除

参数：
- `axis:[int]` 要移除的轴, 默认为[-1]

返回：移除维度为1的变量

返回类型：`Var`

---
###  `round()`

将Var的数据四舍五入

参数：
- `None`

返回：四舍五入的变量

返回类型：`Var`

---
###  `sum(axis)`

返回Var指定轴的和

参数：
- `axis:[int]` 指定的轴, 默认为[-1]

返回：和

返回类型：`Var`

---
###  `var(axis)`

返回Var指定轴的方差

参数：
- `axis:[int]` 指定的轴, 默认为[-1]

返回：标准差

返回类型：`Var`

---
###  `std(axis)`

返回Var指定轴的标准差

参数：
- `axis:[int]` 指定的轴, 默认为[-1]

返回：标准差

返回类型：`Var`

---
###  `swapaxes(axis1, axis2)`

将Var的指定轴交换

参数：
- `axis1:int` 指定的交换轴
- `axis2:int` 指定的交换轴

返回：交换后的变量

返回类型：`Var`

---
###  `transpose(axes)`

将Var转置

参数：
- `axes:[int]` 转置的轴顺序，默认为`None`, 将轴逆序排列

返回：转置后的变量

返回类型：`Var`

---
###  `item(idx)`

返回Var的第idx个元素，类似：`var[idx]`

参数：
- `idx:int` 元素的索引

返回：元素的值

返回类型：`Var`


---
###  `number_overload`
- `+`
- `-`
- `*`
- `/`
- `%`
- `**`
- `abs()`
### `compare_overload`
- `==`
- `!=`
- `<`
- `<=`
- `>`
- `>=`
### `sequence_overload`
- `__iter__`, `__iternext__` 可以通过`for item in var`遍历Var
- `__len__` 返回最高维度的长度
- `__subscript__`
  - `int` 如：`x[0]`
  - `[int]` 如：`x[[0,2]]`
  - `Var` 数据类型为int或bool，如: `x[x > 0]`
  - `PySlice` 如：`x[:, 2], x[0:10:-1]`
- `__ass_subscript__` 与 `__subscript__` 相同，是赋值操作 
### `str_overload`
- `__repr__`, `__str__` 在支持numpy的环境中打印格式与numpy一致，否则以tuple形式打印

### `not_impl_overload`
- `float(var)`
- `max(var1, var2)`
- `if var:`
- *注意：以上均为未定义行为，请勿使用！！！*

---
### `Example`

```python
import MNN.expr as expr
import MNN.numpy as np

# 创建Var变量
# 通过expr.const创建变量，参数分别为：数据，形状，数据排布，数据类型
x = expr.const([1., 2., 3., 4.], [2, 2], F.NCHW, F.float)  # array([[1., 2.], [3., 4.]], dtype=float32)
# 通过numpy.array创建变量，直接传入数据即可
y = np.array([2, 0])  # array([2, 0], dtype=int32)
# 通过numpy.ones创建变量，指定参数为形状
z = np.ones([4]) # array([1, 1, 1, 1], dtype=int32)

# Var的属性
x.shape # [2, 2]
x.size # 4
x.ndim # 2
x.dtype # dtype.float
x.data_format # data_format.NCHW
x.op_type # 'Const'

# 形状变化
x1 = x.reshape(1, 4) # array([[1., 2., 3., 4.]], dtype=float32)
x2 = x.transpose()  # array([[1., 3.], [2., 4.]], dtype=float32)
x3 = x.flatten() # array([1., 2., 3., 4.], dtype=float32)

# 运算符重载
x4 = x3 + z # array([3., 5., 7., 9.], dtype=float32)
x5 = x3 > z # array([0, 1, 1, 1], dtype=int32)

# 数学函数
x.max() # 4.0
x.std() # 1.1180340051651
x.sum() # 10.0

# 元素操作，可以通过索引、切片、迭代等方式进行操作
x[0] # array([1., 2.], dtype=float32)
x[0, 1] # 2.0
x[:, 0] # array([1., 3.], dtype=float32)
x.item(1) # 2.0
len(x) # 2
for item in x:
    print(item) # array([1., 2.], dtype=float32); array([3., 4.], dtype=float32)

# 数据类型转换：到ndarray, tuple, scalar
np_x = x.read() # array([[1., 2.], [3., 4.]], dtype=float32)
type(np_x) # <class 'numpy.ndarray'>
tuple_x = x.read_as_tuple() # (1.0, 2.0, 3.0, 4.0)
type(tuple_x) # <class 'tuple'>
scalar_x = x[0, 1] # 1.0
type(scalar_x) # <class 'float'>
```