## nn

```python
module nn
```
MNN是Pymnn中最基础的Module，其中包含了V2 API所需要数据结构与一些枚举类型；同时包含了一些基础函数。
其他子模块则也需要通过MNN模块引入，引入方式为`import MNN.{module} as {module}`

---
### `nn submodules`
- [loss](loss.md)
- [compress](compress.md)

---
### `nn Types`
- [_Module](_Module.md)
- [RuntimeManager](RuntimeManager.md)

---
### `load_module(inputs, outputs, for_training)`
模块加载类, 将计算图的部分加载为一个`_Module`

参数：
- `inputs:[Var]` 计算图的开始部分
- `outputs:[Var]` 计算图的结束部分
- `for_training:bool` 是否加载为训练模式

返回：模型

返回类型：`_Module`

示例

```python
>>> var_dict = expr.load_as_dict('mobilenet_v1.mnn')
>>> input_var = var_dict['data']
>>> output_var = var_dict['prob']
>>> nn.load_module([input_var], [output_var], False)
<_Module object at 0x7f97c42068f0>
```
---
### `load_module_from_file(file_name, input_names, output_names, |dynamic, shape_mutable, rearrange, backend, memory_mode, power_mode, precision_mode)`
加载模型，从模型文件加载为`_Module`

参数：
- `file_name:str`  模型文件名
- `input_names:list[str]` 输入变量名列表
- `output_names:list[str]` 输出变量名列表
- `dynamic:bool` 是否动态图，默认为False
- `shape_mutable:bool` 是否在内部控制流中形状变化，默认为False
- `rearrange:bool` 是否重新排列输入变量，默认为False
- `backend:expr.Backend` 后端，默认为`expr.Backend.CPU`
- `memory_mode:expr.MemoryMode` 内存模式，默认为`expr.MemoryMode.Normal`
- `power_mode:expr.PowerMode` 功耗模式，默认为`expr.PowerMode.Normal`
- `precision_mode:expr.PrecisionMode` 精度模式，默认为`expr.PrecisionMode.Normal`
- `thread_num:int` 使用线程数，默认为`4`

返回：创建的模型

返回类型：`_Module`

示例

```python
>>> conv = nn.conv(3, 16, [3, 3])
>>> input_var = np.random.random((1, 3, 64, 64))
>>> conv(input_var)
array([[[[-5.25599599e-01,  3.63697767e-01,  5.57627320e-01, ...,
          -3.90964895e-01, -3.85326982e-01,  5.49694777e-01],
          ...,
          [-8.73677015e-01,  2.95535415e-01,  3.95657867e-02, ...,
           5.87978542e-01, -1.16958594e+00,  1.74816132e-01]]]], dtype=float32)
```
---
### `create_runtime_manager(config)`
根据config信息创建[RuntimeManager](RuntimeManager.md)

参数：
- `config:str` 配置信息，参考[createRuntime](Interpreter.html#createruntime-config)

返回：创建的[RuntimeManager](RuntimeManager.md)

返回类型：`RuntimeManager`

---
### `conv(in_channel, out_channel, kernel_size, stride, padding, dilation, depthwise, bias, padding_mode)`
创建卷积模块实例

参数：
- `model_path:str` 模型路径
- `in_channel:int` 输入通道数
- `out_channel:int` 输出通道数
- `kernel_size:int` 卷积核大小
- `stride:list[int]` 卷积步长，默认为[1, 1]
- `padding:list[int]` 填充大小，默认为[0, 0]
- `dilation:list[int]` 卷积核膨胀，默认为[1, 1]
- `depthwise:bool` 是否深度卷积，默认为False
- `bias:bool` 是否使用偏置，默认为True
- `padding_mode:expr.Padding_Mode` 填充模式，默认为`expr.Padding_Mode.VALID`

返回：卷积模块

返回类型：`_Module`

示例

```python
>>> conv = nn.conv(3, 16, [3, 3])
>>> input_var = np.random.random((1, 3, 64, 64))
>>> conv(input_var)
array([[[[-5.25599599e-01,  3.63697767e-01,  5.57627320e-01, ...,
          -3.90964895e-01, -3.85326982e-01,  5.49694777e-01],
          ...,
          [-8.73677015e-01,  2.95535415e-01,  3.95657867e-02, ...,
           5.87978542e-01, -1.16958594e+00,  1.74816132e-01]]]], dtype=float32)
```
---
### `linear(input_length, output_length, bias)`
创建innerproduct实例

参数：
- `input_length:int` 输入长度
- `output_length:int` 输出长度
- `bias:bool` 是否使用偏置，默认为True

返回：线性模块

返回类型：`_Module`

示例

```python
>>> linear = nn.linear(32, 64)
>>> input_var = np.random.random([32])
>>> linear(input_var)
```

---
### `batch_norm(channels, dims, momentum, epsilon)`
创建batchnorm实例

参数：
- `channels:int` 通道数
- `dims:int` 维度，默认为4
- `momentum:float` 动量，默认为0.99
- `epsilon:float` 极小值，默认为1e-05

返回：batch_norm模块

返回类型：`_Module`

示例

```python
>>> bn = nn.batch_norm(3)
>>> input_var = np.random.random([1, 3, 2, 2])
>>> bn(input_var)
array([[[[-1.5445713 ,  1.0175514 ],
         [-0.20512265,  0.73214275]],
        [[-0.9263869 , -0.59447914],
         [-0.14278792,  1.663654  ]],
        [[-0.61769044,  0.15747389],
         [-1.0898823 ,  1.5500988 ]]]], dtype=float32)
```

---
### `dropout(drop_ratio)`
创建dropout实例

参数：
- `drop_ratio:float` dropout比例

返回：dropout模块

返回类型：`_Module`

示例

```python
>>> dropout = nn.dropout(0.5)
>>> input_var = np.random.random([8])
>>> dropout(input_var)
array([0.0000000e+00, 1.9943696e+00, 1.4406490e+00, 0.0000000e+00,
       2.2876216e-04, 0.0000000e+00, 6.0466516e-01, 1.9980811e+00], dtype=float32)
```
