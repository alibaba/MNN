## MNN.Interpreter *[deprecated]*

```python
class Interpreter
```
Interpreter是MNN V2接口中模型数据的持有者。使用MNN推理时，有两个层级的抽象，分别是解释器Interpreter和会话[Session](Session.md)。

*不建议使用该接口，请使用[nn](nn.md)代替*

---
### `Interpreter(model_path)`
加载`.mnn`模型文件创建一个MNN解释器，返回一个解释器对象

参数：
- `model_path:str` MNN模型所放置的完整文件路径，其中MNN模型可由Tensorflow、Caffe、PyTorch和 ONNX等模型进行转换得到

返回：Interpreter对象

返回类型：`Interpreter`

---
### `createRuntime(config)`

根据配置创建一个Runtime，并获取config中指定的参数是否生效；默认情况下，在`createSession`时对应create单独一个Runtime。对于串行的一系列模型，可以先单独创建Runtime，然后在各Session创建时传入，使各模型用共享同样的运行时资源（对CPU而言为线程池、内存池，对GPU而言Kernel池等），参考[RuntimeManager](RuntimeManager.md)

参数：
- `config:dict` 创建Runtime的配置, 其key, value和含义如下变所示

|    key        |  value  |      说明         |
|:--------------|:--------|:-----------------|
| backend     | `str` or `int` | 可选值：`"CPU"或0`(默认), `"OPENCL"或3`,`"OPENGL"或6`, `"VULKAN"或7`, `"METAL"或1`, `"TRT"或9`, `"CUDA"或2`, `"HIAI"或8`  |
| precision   | `str` | 可选值：`"normal"`(默认), `"low"`,`"high","lowBF"` |
| numThread   | `int` or `long` | `value`为推理线程数，只在 CPU 后端下起作用 |
| saveTensors | `tuple` of `str` | `value`为想要保留成为输出层的`tensorName` |
| inputPaths  | `tuple` of `str` | 推理路径的起点，输入`tensorName` |
| outputPaths | `tuple` of `str` | 推理路径的终点，输出`tensorName` |

返回：一个`pair`，
- first：Runtime对象的`PyCapsule`，可以用来创建Session
- second：为`tuple` of `bool`；代表config中对应的配置是否生效

返回类型：`pair`

---
### `createSession(config, |runtime)`

根据配置创建[Session](Session.md)，返回一个`Session`对象。

参数：
- `config:dict` 创建推理会话的配置，含义同[createRuntime](Interpreter.html#createruntime-config)方法
- `runtime:PyCapsule` 指定的runtime信息，如果不指定，则使用config中的配置创建runtime

返回：持有推理会话数据的Session对象

返回类型：`Session`

---
### `setCacheFile(cache_path)`

设置缓存文件路径，在GPU情况下可以把kernel和Op-info缓存到该文件中

参数：
- `cache_path:str` 缓存文件的路径

返回：`None`

返回类型：`None`

---
### `setExternalFile(path)`

设置额外数据文件路径，使用该文件中的数据作为权重或常量

参数：
- `path:str` 额外数据文件的路径

返回：`None`

返回类型：`None`

---
### `updateCacheFile(session, flag)`

在执行推理之后，更新GPU的kernel信息到缓存文件；应该在每次推理结束后指定该函数

参数：
- `session:Session` 需要缓存的会话
- `flag` 保留参数，目前未使用；输入`0`即可

返回：error code 参考`runSession`方法

返回类型：`int`

---
### `setSessionMode(mode)`

设置会话的执行模式

参数：
- `mode:int` 执行Session的模式，含义如下表所示

| value | name |      说明         |
|:------|:-----|:-----------------|
| 0 | `Session_Debug` | 可以执行callback函数，并获取Op信息(*默认*) |
| 1 | `Session_Release` | 不可执行callback函数 |
| 2 | `Session_Input_Inside` | 输入由session申请(*默认*) |
| 3 | `Session_Input_User` | 输入由用户申请 |
| 4 | `Session_Output_Inside` | 输出依赖于session不可单独使用 |
| 5 | `Session_Output_User` | 输出不依赖于session可单独使用 |
| 6 | `Session_Resize_Direct` | 在创建Session时执行resize(*默认*) |
| 7 | `Session_Resize_Defer` | 在创建Session时不执行resize |
| 8 | `Session_Backend_Fix` | 使用用户指定的后端，后端不支持时回退CPU |
| 9 | `Session_Backend_Auto` | 根据算子类型自动选择后端 |

返回：`None`

返回类型：`None`

---
### `setSessionHint(mode, value)`

设置执行时的额外信息

参数：
- `mode:int` hint类型
- `value:int` hint值

| mode | name |      说明         |
|:-----|:-----|:-----------------|
| 0 | `MAX_TUNING_NUMBER` | GPU下tuning的最大OP数 |

返回：`None`

返回类型：`None`

---
### `getSessionInput(session, |tensorName)`

根据tensorName，返回模型指定会话的输入tensor；如果没有指定tensor名称，则返回第一个输入tensor

参数：
- `session:Session` 持有推理会话数据的Session对象
- `tensorName:str` Tensor的名称

返回：输入Tensor对象

返回类型：`Tensor`

---
### `getSessionInputAll(session)`

返回模型指定会话的所有的输入tensor

参数：
- `session:Session` 持有推理会话数据的Session对象

返回：所有的输入Tensor对象，类型为字典，其中key为tensorName，类型为str；value为一个输入tensor，类型为Tensor 

返回类型：`dict`

---
### `getSessionOutput(session, |tensorName)`

根据tensorName，返回模型指定会话的输出tensor；如果没有指定tensor名称，则返回第一个输出tensor

参数：
- `session:Session` 持有推理会话数据的Session对象
- `tensorName:str` Tensor的名称

返回：输出Tensor对象

返回类型：`Tensor`

---
### `getSessionOutputAll(session)`

返回模型指定会话的所有的输出tensor

参数：
- `session:Session` 持有推理会话数据的Session对象

返回：所有的输出Tensor对象，类型为字典，其中key为tensorName，类型为str；value为一个输入tensor，类型为Tensor 

返回类型：`dict`

---
### `resizeTensor(tensor, shape)`

改变tensor形状，并重新分配内存

参数：
- `tensor:Tensor` 需要改变形状的Tensor对象，一般为输入tensor
- `shape:tuple` 新的形状，元素类型为`int`

返回：`None`

返回类型：`None`

---
### `resizeSession(session)`

为session分配内存，进行推理准备工作；该API一般配合`resizeTensor`一起调用，修改Tensor输入形状后对应整个推理过程中的内存分配也需要修改。

参数：
- `session:Session` 改变输入形状后需要重新分配内存的Session对象

返回：成功返回True，否则抛出相应的异常

返回类型：`bool`

---
### `runSession(session)`

运行session执行模型推理，返回对应的error code，需要根据错误码来判断后续是否成功执行模型推理

参数：
- `session:Session` 执行推理的Session对象

返回：错误码，具体含义如下表

| value | name |      说明         |
|:------|:-----|:-----------------|
| 0 | `NO_ERROR` | 没有错误，执行成功  |
| 1 | `OUT_OF_MEMORY` | 内存不足，无法申请内存 |
| 2 | `NOT_SUPPORT` | 有不支持的OP |
| 3 | `COMPUTE_SIZE_ERROR` | 形状计算出错 |
| 4 | `NO_EXECUTION` | 创建执行时出错 |
| 10 | `INPUT_DATA_ERROR` | 输入数据出错 |
| 11 | `CALL_BACK_STOP` | 用户callback函数退出 |
| 20 | `TENSOR_NOT_SUPPORT` | resize出错 |
| 21 | `TENSOR_NEED_DIVIDE` | resize出错 |

返回类型：`int`

---
### `runSessionWithCallBack(session, begin_callback, end_callback)`

该API本质上与runSession一致，但是提供了用户hook函数的接口，在运行session做网络推理，每层推理前前后会执行的begin_callback和end_callback 并根据返回值来决定是否继续执行

参数：
- `session:Session` 执行推理的Session对象
- `begin_callback:function|lambda` 每层推理前执行的回调函数，函数原型为：
    ```
    def begin_callback(tensors, name):
        # do something
        return True
    ```
    参数：
    - `tensors:[Tensor]` 该层的输入tensor
    - `name:str` 该层的名称 
    返回：`True`继续执行推理，`False`停止执行推理
    返回类型：`bool`
- `end_callback:function|lambda` 每层推理后执行的回调函数，函数原型同上

返回：同runSession

返回类型：`int`

---
### `runSessionWithCallBackInfo(session, begin_callback, end_callback)`

该API与runSessionWithCallBack相似，但是回调函数中增加了Op的类型和计算量信息，可以用来评估模型的计算量

参数：
- `session:Session` 执行推理的Session对象
- `begin_callback:function|lambda` 每层推理前执行的回调函数，函数原型为：
    ```
    def begin_callback(tensors, opinfo):
        # do something
        return True
    ```
    参数：
    - `tensors:[Tensor]` 该层的输入tensor
    - `opinfo:OpInfo` 该层Op的相关信息，参考[OpInfo](OpInfo.md)
    返回：`True`继续执行推理，`False`停止执行推理
    返回类型：`bool`
- `end_callback:function|lambda` 每层推理后执行的回调函数，函数原型同上

返回：同runSession

返回类型：`int`

---
### `cache()`

将该Interpreter存储到当前线程的缓存中，以便多次使

参数：
- `None`

返回：`None`

返回类型：`None`

---
### `removeCache()`

将该Session从当前线程的缓存中移除

参数：
- `None`

返回：`None`

返回类型：`None`

---
### `Example`
    
```python
import MNN
import MNN.cv as cv
import MNN.numpy as np

# 创建interpreter
interpreter = MNN.Interpreter("mobilenet_v1.mnn")
# 创建session
session = interpreter.createSession()
# 获取会话的输入输出张量
input_tensor = interpreter.getSessionInput(session)
output_tensor = interpreter.getSessionOutput(session)
# 将输入resize到[1, 3, 224, 224]
interpreter.resizeTensor(input_tensor, (1, 3, 224, 224))

# 读取图片，转换为size=(224, 224), dtype=float32，并赋值给input_tensor
image = cv.imread('cat.jpg')
image = cv.resize(image, (224, 224), mean=[103.94, 116.78, 123.68], norm=[0.017, 0.017, 0.017])
# HWC to NHWC
image = np.expand_dims(image, 0)
tmp_input = MNN.Tensor(image)
input_tensor.copyFrom(tmp_input)

# 执行会话推理
interpreter.runSession(session)

# 将输出结果拷贝到tmp_output中
tmp_output = MNN.Tensor((1, 1001), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
output_tensor.copyToHostTensor(tmp_output)

# 打印出分类结果, 282为猫
print("output belong to class: {}".format(np.argmax(tmp_output.getData())))
```