<!-- pymnn/Tensor.md -->
## MNN.Tensor *[deprecated]*

```python
class Tensor
```
Tensor是MNN V2接口中的基础数据结构，是最基本的数据封装类型。Tensor存储了数据以及数据类型，形状等诸多信息，用户可以通过Tensor本身的函数获取这些信息。

*不建议使用该接口，请使用[Var](Var.md)代替*

---
### `MNN.Halide_Type_*`
描述Tensor的数据类型
- 类型：`PyCapsule`
- 枚举值：
  - `Halide_Type_Double`
  - `Halide_Type_Float`
  - `Halide_Type_Int`
  - `Halide_Type_Int64`
  - `Halide_Type_String`
  - `Halide_Type_Uint8 `

---
### `MNN.Tensor_DimensionType_*`
描述Tensor的数据排布格式
- 类型：`int`
- 枚举值：
  - `Tensor_DimensionType_Caffe`
  - `Tensor_DimensionType_Caffe_C4`
  - `Tensor_DimensionType_Tensorflow`

---
### `Tensor()`
创建一个空Tensor

参数：
- `None`

返回：Tensor对象

返回类型：`Tensor`

### `Tensor(var)`

创建一个Tensor，并使用var的数据

参数：
- `var:Var` 类型为Var的变量

### `Tensor(tensor_or_var, dimension)`
创建一个Tensor，并使用tensor_or_var的数据, 并将数据排布格式设置为dimension

参数：
- `tensor_or_var:Tensor/Var` 类型为Tensor或者Var的变量
- `dimension:MNN.Tensor_DimensionType_*` 数据排布格式

### `Tensor(shape, dtype, dimension)`
创建一个指定形状，数据类型和数据排布的Tensor, 数据未经初始化

参数：
- `shape:tuple` Tensor形状
- `dtype:MNN.Halide_Type_*` Tensor数据类型
- `dimension:MNN.Tensor_DimensionType_*` 数据排布格式

### `Tensor(shape, dtype, value_list, dimension)`
创建一个指定形状，数据类型, 数据和数据排布的Tensor, 数据拷贝自`value_list`，
能够将`list`，`tuple`，`bytes`，`ndarray`，`PyCapsule`，`int指针`等格式的数据转换成`Tensor`

*注意：`value_list`仅在PYMNN_NUMPY_USABLE打开的情况下支持`ndarray`，移动端默认关闭*

*此函数在`PYMNN_NUMPY_USABLE=OFF`时不接受`ndarray`作为数据输入*

参数：
- `shape:tuple` Tensor形状
- `dtype:MNN.Halide_Type_*` Tensor数据类型
- `value_list:ndarray/tuple/list/bytes/PyCapsule/int_addr` 数据
- `dimension:MNN.Tensor_DimensionType_*` 数据排布格式

---
### `getShape()`

获取Tensor的形状。

参数：
- `None`

返回：Tensor的数据形状

返回类型：`Tuple`

---
### `getDataType()`

获取Tensor的数据类型。

参数：
- `None`

返回：Tensor的数据类型

返回类型：`MNN.Halide_Type_*`

---
### `getDimensionType()`

获取Tensor的持有的数据排布格式。

参数：
- `None`

返回：`Tensor`持有的数据排布格式

返回类型：`MNN.Tensor_DimensionType_*`

---
### `getData()`

获取Tensor的数据。

参数：
- `None`

返回：Tensor的数据

返回类型：`Tuple`

---
### `getHost()`

获取Tensor的数据指针。

参数：
- `None`

返回：Tensor内部数据的数据指针

返回类型：`PyCapsule`, 可以参考[PyCapsule介绍](https://docs.python.org/3/c-api/capsule.html)

---
### `copyFrom(from)`

从from中拷贝数据到当前Tensor，可用此函数将数据拷贝到输入Tensor中。

参数：
- `from:Tensor` - 拷贝的源Tensor

返回：是否拷贝成功

返回类型：`bool`

---
### `copyToHostTensor(to)`

从当前Tensor拷贝数据到to，可用此函数将输出Tensor中的数据拷出。

参数：
- `to:Tensor` - 拷贝的目标Tensor

返回：是否拷贝成功

返回类型：`bool`

---
### `getNumpyData()`

获取Tensor的数据，返回numpy数据。
*该API仅在PYMNN_NUMPY_USABLE=ON时生效，移动端默认关闭*

参数：
- `None`

返回：Tensor数据的numpy形式

返回类型：`numpy.ndarray`


---
### `Example`
    
```python
import numpy as _np
import MNN
import MNN.numpy as np
data = _np.array([1., 2., 3.], dtype=_np.float32)
# 创建Tensor
# 通过给定的tuple创建Tensor, 参数分别为：形状，数据类型，数据，数据排布格式
t1 = MNN.Tensor((1, 3), MNN.Halide_Type_Float, (1., 2., 3.), MNN.Tensor_DimensionType_Caffe)
# 通过Var创建Tensor
t2 = MNN.Tensor(np.array([1., 2., 3.])) # 与t1等价
# 通过ndarray创建Tensor
t3 = MNN.Tensor([1, 3], MNN.Halide_Type_Float, data, MNN.Tensor_DimensionType_Caffe)
# 通过bytes创建Tensor
t4 = MNN.Tensor([1, 3], MNN.Halide_Type_Float, data.tobytes(), MNN.Tensor_DimensionType_Caffe)
# 通过int类型的内存指针创建Tensor，使用该方法比直接用ndarray速度快，但是要求ndarray的内存必须连续
t5 = MNN.Tensor([1, 3], MNN.Halide_Type_Float, data.__array_interface__['data'][0], MNN.Tensor_DimensionType_Caffe)

print(t1.getShape()) # (1, 3)
print(t1.getDataType()) # <capsule object NULL at 0x7fe01e74ff30>
print(t1.getDimensionType()) # 1
print(t1.getData()) # (1.0, 2.0, 3.0)
print(t1.getHost()) # <capsule object NULL at 0x7fe01e5645a0>
print(t1.getNumpyData()) # [[1. 2. 3.]]
```